import glob
import itertools
import os
import subprocess
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from pycosep.concorde_settings import ConcordeSettings
from pycosep.separability_variants import SeparabilityVariant


def _mode_distribution(data_clustered):
    mode_dist = np.empty([0])
    _, dims = data_clustered.shape

    for ix in range(dims):
        kde = stats.gaussian_kde(data_clustered[:, ix])
        xi = np.linspace(data_clustered.min(), data_clustered.max(), 100)
        p = kde(xi)
        ind = np.argmax([p])
        mode_dist = np.append(mode_dist, xi[ind])

    return mode_dist


def _find_positive_classes(sample_labels):
    positives, positions = np.unique(sample_labels, return_inverse=True)
    max_pos = np.bincount(positions).argmax()
    positives = np.delete(positives, max_pos)

    return positives


def _extract_positive_class(communities_membership, positives):
    positive_community_class = None
    for o in range(len(positives)):
        if np.any(communities_membership == positives[o]):
            positive_community_class = positives[o]
            break

    if positive_community_class is None:
        raise RuntimeError('impossible to set the current positive community class')

    return positive_community_class


def _compute_mann_whitney(scores_c1, scores_c2):
    mw = stats.mannwhitneyu(scores_c1, scores_c2)  # method="exact"
    return mw


def _compute_auc_aupr(labels, scores, positives):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=positives)
    auc = metrics.auc(fpr, tpr)

    if auc < 0.5:
        auc = 1 - auc
        flipped_scores = 2 * np.mean(scores) - scores
        precision, recall, thresholds = metrics.precision_recall_curve(labels, flipped_scores, pos_label=positives)
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=positives)

    aupr = metrics.auc(recall, precision)

    return auc, aupr


def _compute_mcc(labels, scores, positives):
    total_positive = np.sum(labels == positives)
    total_negative = np.sum(labels != positives)
    negative_class = np.unique(labels[labels != positives]).item()
    true_labels = labels[np.argsort(scores)]

    ps = np.array([positives] * total_positive)
    ng = np.array([negative_class] * total_negative)

    coefficients = np.empty([0])
    for ix in range(0, 2):
        if ix == 0:
            predicted_labels = np.concatenate((ps, ng), axis=0)
        else:
            predicted_labels = np.concatenate((ng, ps), axis=0)
        coefficients = np.append(coefficients, metrics.matthews_corrcoef(true_labels, predicted_labels))

    mcc = np.max(coefficients)

    return mcc


def _create_line_between_centroids(centroid1, centroid2):
    line = np.vstack([centroid1, centroid2])
    return line


def _project_point_on_line(point, line):
    # centroids
    a = line[0]
    b = line[1]

    # deltas
    ap = point - a
    ab = b - a

    # projection
    projected_point = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

    return projected_point


def _convert_points_to_one_dimension(points):
    start_point = None
    _, dims = points.shape

    for ix in range(dims):
        if np.unique(points[:, ix]).size != 1:
            start_point = np.array(points[np.argmin(points[:, ix], axis=0), :]).reshape(1, dims)
            break

    if start_point is None:
        raise RuntimeError('impossible to set projection starting point')

    v = np.zeros(np.shape(points)[0])
    for ix in range(dims):
        v = np.add(v, np.power(points[:, ix] - np.min(start_point[:, ix]), 2))

    v = np.sqrt(v)

    return v


def _convert_tour_to_one_dimension(best_tour, pairwise_data, pairwise_communities):
    input_nodes = np.array(best_tour)
    target_nodes = np.roll(input_nodes, -1)

    # compute weights (Euclidean distances)
    weights = np.zeros(len(input_nodes))
    for ix in range(len(input_nodes)):
        weights[ix] = np.linalg.norm(pairwise_data[input_nodes[ix], :] - pairwise_data[target_nodes[ix], :])

    # Create weighted graph using Euclidean distances
    tour_graph = nx.Graph()
    for ix in range(len(input_nodes)):
        tour_graph.add_edge(int(input_nodes[ix]), int(target_nodes[ix]), weight=weights[ix])

    start_node = None
    end_node = None

    # Sort edges by weight in descending order
    edges_sorted = sorted(tour_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    for edge in edges_sorted:
        node_a, node_b, data = edge
        community_a = pairwise_communities[node_a]
        community_b = pairwise_communities[node_b]

        if community_a != community_b:
            start_node = node_a
            end_node = node_b
            break

    if start_node is None or end_node is None:
        raise RuntimeError('cannot find best tour cut')

    # create TSP projection path: remove the longest edge that splits the communities
    tour_graph.remove_edge(start_node, end_node)

    # compute the shortest path on the TSP projection path
    s_path = nx.shortest_path(tour_graph, source=start_node, target=end_node, weight='weight')

    # compute scores: distance from each node to start_node on the TSP projection path
    scores = np.zeros(len(s_path))
    for ix, node in enumerate(s_path):
        try:
            distance = nx.shortest_path_length(tour_graph, source=node, target=start_node, weight='weight')
            scores[node] = distance
        except nx.NetworkXNoPath:
            scores[node] = 0

    return scores, input_nodes, target_nodes, weights, start_node, end_node


def _centroid_based_projection(data_group_a, data_group_b, center_formula):
    if center_formula != 'mean' and center_formula != 'median' and center_formula != 'mode':
        warnings.warn('invalid center formula: median will be applied by default', SyntaxWarning)
        center_formula = 'median'

    centroid_a = centroid_b = None
    if center_formula == 'median':
        centroid_a = np.median(data_group_a, axis=0)
        centroid_b = np.median(data_group_b, axis=0)
    elif center_formula == 'mean':
        centroid_a = np.mean(data_group_a, axis=0)
        centroid_b = np.mean(data_group_b, axis=0)
    elif center_formula == 'mode':
        centroid_a = _mode_distribution(data_group_a)
        centroid_b = _mode_distribution(data_group_b)

    if centroid_a is None or centroid_b is None:
        raise RuntimeError('impossible to set clusters centroids')
    elif np.array_equal(centroid_a, centroid_b):
        raise RuntimeError('clusters have the same centroid: no line can be traced between them')

    centroids_line = _create_line_between_centroids(centroid_a, centroid_b)
    pairwise_data = np.vstack([data_group_a, data_group_b])

    total_points, total_dimensions = np.shape(pairwise_data)
    projection = np.empty([0, total_dimensions])
    for ox in range(total_points):
        projected_point = _project_point_on_line(pairwise_data[ox], centroids_line)
        projection = np.vstack([projection, projected_point])

    return projection


def _lda_based_projection(pairwise_data, pairwise_samples):
    mdl = LinearDiscriminantAnalysis(solver='svd', store_covariance=True, n_components=1)
    mdl.fit(pairwise_data, pairwise_samples)
    mu = np.mean(pairwise_data, axis=0)
    # projecting data points onto the first discriminant axis
    centered = pairwise_data - mu
    projection = np.dot(centered, mdl.scalings_ * np.transpose(mdl.scalings_))
    projection = projection + mu

    return projection


def _tsp_based_projection(pairwise_data, concorde_settings):
    # check if 'Concorde' path is specified in runtime settings
    if concorde_settings.concorde_path == '':
        raise ValueError("'Concorde' path not specified in runtime settings")

    # sanity check to avoid overriding TSP input and output files
    while True:
        short_uuid = str(uuid.uuid4())[-12:].replace('-', '')

        file_name = short_uuid
        file_path = os.path.join(concorde_settings.temp_path, file_name)

        file_tsp_name = file_name + '.tsp'
        file_tsp_path = os.path.join(concorde_settings.temp_path, file_tsp_name)
        file_sol_name = file_name + '.sol'
        file_sol_path = os.path.join(concorde_settings.temp_path, file_sol_name)

        if not (os.path.isfile(file_tsp_path) or os.path.isfile(file_sol_path)):
            break
        else:
            warnings.warn(f"TSP or Solution file '{file_path}' already exists. Retrying...", RuntimeWarning)
            time.sleep(0.25)

    total_nodes = pairwise_data.shape[0]

    # sanity check to avoid "rounding-up problem" when using EUC_2D
    # see: https://stackoverflow.com/questions/27304721/how-to-improve-the-quality-of-the-concorde-tsp-solver-am-i-misusing-it
    abs_mean = abs(np.median(pairwise_data))
    digits = 3
    inverted_magnitude = 10 ** (2 - digits + np.floor(np.log10(abs_mean)))
    offset = 1
    while True:
        scaling_factor = 10 ** (abs(np.log10(inverted_magnitude)) + offset)
        scaled_embedding = pairwise_data * scaling_factor
        max_value = np.max(scaled_embedding)
        max_distance = np.max(squareform(pdist(scaled_embedding)))
        if max_value < np.iinfo(np.int32).max and max_distance < 32768:
            break
        offset -= 1

    # prepare TSP file
    with open(file_tsp_path, 'w') as file:
        file.write("NAME : TSPS Concorde\n")
        file.write(f"COMMENT : Scaling factor {scaling_factor}\n")
        file.write("TYPE : TSP\n")
        file.write(f"DIMENSION : {total_nodes}\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")
        for ix in range(total_nodes):
            file.write(f"{ix + 1} {scaled_embedding[ix, 0]} {scaled_embedding[ix, 1]}\n")
        file.write("EOF\n")

    # execute Concorde
    command = f"{concorde_settings.concorde_path} -s 40 -x -o {file_sol_name} {file_tsp_name}"

    result = subprocess.run(command, cwd=concorde_settings.temp_path, capture_output=True, text=True, check=False,
                            shell=True)
    if result.returncode != 0 and result.returncode != 255:
        raise RuntimeError(f"Error executing Concorde command:\t\n{result.stdout}\t\n{result.stderr}")

    # Process Concorde's solution file
    try:
        with open(file_sol_path, 'r') as file:
            # skip the first element (number of nodes)
            lines = file.readlines()[1:]

        loaded_tour = []
        for line in lines:
            nodes = [int(x) for x in line.split()]
            loaded_tour.extend(nodes)

        best_route = np.array(loaded_tour)
    except Exception as e:
        warnings.warn(f"Cannot process Concorde solution: {e}", RuntimeWarning)
        best_route = np.array([])

    # clean up Concorde's related files
    if os.path.isfile(file_tsp_path):
        # sometimes, Concorde creates more than just the TSP and Solution file
        # hence, we clean up all files associated to the computed tour
        pattern = os.path.join(concorde_settings.temp_path, f"{file_name}.*")
        associated_files = glob.glob(pattern)

        for associated_file in associated_files:
            os.remove(associated_file)

    return best_route


def _randomize_communities(communities, total_permutations):
    randomized = []
    total_communities = len(communities)

    for ix in range(total_permutations):
        np.random.seed(ix)
        positions = np.random.permutation(total_communities)
        randomized.append(communities[positions])

    return randomized


def _compute_for_permutation(jx, permuted_communities_membership, scores, current_positive_class):
    permuted_communities = permuted_communities_membership[jx]
    auc, aupr = _compute_auc_aupr(permuted_communities, scores, current_positive_class)
    mcc = _compute_mcc(permuted_communities, scores, current_positive_class)
    return auc, aupr, mcc


def _compute_separability_measures(permuted_communities_membership, scores, current_positive_class, total_permutations):
    auc_values = np.zeros(total_permutations)
    aupr_values = np.zeros(total_permutations)
    mcc_values = np.zeros(total_permutations)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            _compute_for_permutation,
            range(total_permutations),
            [permuted_communities_membership] * total_permutations,
            [scores] * total_permutations,
            [current_positive_class] * total_permutations
        ))

    for ix, (auc, aupr, mcc) in enumerate(results):
        auc_values[ix] = auc
        aupr_values[ix] = aupr
        mcc_values[ix] = mcc

    return auc_values, aupr_values, mcc_values


def _set_permutation_results(permuted_values, original_value, total_permutations):
    permutations = dict(
        original_value=original_value,
        permutations=permuted_values,
        p_value=(np.sum(permuted_values >= original_value) + 1) / (total_permutations + 1),
        mean=np.mean(permuted_values),
        max=np.max(permuted_values),
        min=np.min(permuted_values),
        standard_deviation=np.std(permuted_values),
        standard_error=np.std(permuted_values) / np.sqrt(total_permutations),
    )

    return permutations


def _compute_permutations(previous_results, metadata, communities, positives, total_permutations):
    total_pairwise_combinations = len(metadata)

    auc_values = np.zeros((total_permutations, total_pairwise_combinations))
    aupr_values = np.zeros((total_permutations, total_pairwise_combinations))
    mcc_values = np.zeros((total_permutations, total_pairwise_combinations))

    for ix in range(total_pairwise_combinations):
        meta = metadata[ix]

        communities_group_a = communities[np.isin(communities, meta['community_name_group_a'])]
        communities_group_b = communities[np.isin(communities, meta['community_name_group_b'])]

        communities_membership = np.concatenate([communities_group_a, communities_group_b])
        permuted_communities_membership = _randomize_communities(communities_membership, total_permutations)
        current_positive_class = _extract_positive_class(communities_membership, positives)

        scores = meta['scores']

        auc_values[:, ix], aupr_values[:, ix], mcc_values[:, ix] = _compute_separability_measures(
            permuted_communities_membership, scores, current_positive_class, total_permutations
        )

    permutations = {}

    corrected_auc_values = np.mean(auc_values, axis=1) / (1 + np.std(auc_values, axis=1))
    auc_original_variant_value = previous_results['auc']
    permutations['auc'] = _set_permutation_results(corrected_auc_values, auc_original_variant_value,
                                                   total_permutations)

    corrected_aupr_values = np.mean(aupr_values, axis=1) / (1 + np.std(aupr_values, axis=1))
    aupr_original_variant_value = previous_results['aupr']
    permutations['aupr'] = _set_permutation_results(corrected_aupr_values, aupr_original_variant_value,
                                                    total_permutations)

    corrected_mcc_values = np.mean(mcc_values, axis=1) / (1 + np.std(mcc_values, axis=1))
    mcc_original_variant_value = previous_results['mcc']
    permutations['mcc'] = _set_permutation_results(corrected_mcc_values, mcc_original_variant_value,
                                                   total_permutations)

    return permutations


def compute_separability(embedding, communities, positives=None, variant=SeparabilityVariant.CPS, permutations=None,
                         concorde_settings=None):
    """
    Compute all community separability indices

    :param embedding: numpy.ndarray
        Data in the form of an N*M matrix where sample values are placed in the rows and
        the feature/variable values are placed in the columns. For instance, this is the output obtained
        after applying a network embedding algorithm.
    :param communities: numpy.ndarray
        List of community labels (e.g., ground truth groups/classes) of the data.
    :param positives: numpy.ndarray
        List of positive community labels. Depending on the study, positive classes are usually ranked as
        the labels for which a particular prediction is desired.
        For instance:
            - sick patients (positive class) versus controls (negative class)
            - burnout (positive class), depression (positive class), versus control (negative class)
        If not provided, then the communities with the lower number of samples will be selected as
        positive classes.
    :param variant: SeparabilityVariant
        Community separability variant to use for computing the separability. This is one of:
            - CPS: centroid projection separability
            - LDPS: linear discriminant projection separability
            - TSPS: travelling salesman projection separability
    :param permutations: int
        Number of iterations for the null model.
    :param concorde_settings: ConcordeSettings
        Concorde settings to use at runtime. For instance, path to temp artifacts and to
        Concorde executable. Only required when using the TSPS variant
    :return:
        indices: dict
            Dictionary containing all the computed community separability indices.
        metadata: dict
            Dictionary containing the metadata used to compute community separability indices.
    """

    # sanity checks
    if type(embedding) is not np.ndarray:
        raise TypeError("invalid input type: 'embedding' must be a numpy.ndarray")

    if type(communities) is not np.ndarray:
        raise TypeError("invalid input type: 'communities' must be a numpy.ndarray")

    if positives is None:
        positives = _find_positive_classes(communities)
    elif type(positives) is not np.ndarray:
        raise TypeError("invalid input type: 'positives' must be a numpy.ndarray")

    if not isinstance(variant, SeparabilityVariant):
        warnings.warn(f"invalid separability variant '{variant}': 'cps' will be used by default", SyntaxWarning)
        variant = SeparabilityVariant.CPS

    # check range of dimensions
    total_samples, total_dimensions = embedding.shape
    if len(communities) != total_samples:
        raise IndexError("the number of 'communities' does not match the number of rows in the provided 'embedding'")

    # extract communities
    unique_communities = np.unique(communities)
    total_communities = len(unique_communities)

    # segregate data according to extracted communities
    communities_clustered = list()
    data_clustered = list()
    for k in range(total_communities):
        idxes = np.where(communities == unique_communities[k])
        communities_clustered.append(communities[idxes])
        data_clustered.append(embedding[idxes])

    auc_values = np.empty([0])
    aupr_values = np.empty([0])
    mcc_values = np.empty([0])

    pairwise_group_combinations = list(itertools.combinations(range(0, total_communities), 2))
    total_pairwise_group_combinations = len(pairwise_group_combinations)

    metadata = [{} for _ in range(total_pairwise_group_combinations)]

    for index_group_combination in range(total_pairwise_group_combinations):
        index_group_a = pairwise_group_combinations[index_group_combination][0]
        data_group_a = data_clustered[index_group_a]
        communities_group_a = communities_clustered[index_group_a]
        community_name_group_a = unique_communities[index_group_a]
        metadata[index_group_combination]["community_name_group_a"] = community_name_group_a
        metadata[index_group_combination]["data_group_a"] = data_group_a

        index_group_b = pairwise_group_combinations[index_group_combination][1]
        data_group_b = data_clustered[index_group_b]
        communities_group_b = communities_clustered[index_group_b]
        community_name_group_b = unique_communities[index_group_b]
        metadata[index_group_combination]["community_name_group_b"] = community_name_group_b
        metadata[index_group_combination]["data_group_b"] = data_group_b

        scores = None
        if variant == SeparabilityVariant.CPS:
            center_formula = 'median'
            projected_points = _centroid_based_projection(data_group_a, data_group_b, center_formula)
            if not projected_points.size == 0:
                scores = _convert_points_to_one_dimension(projected_points)
        elif variant == SeparabilityVariant.LDPS:
            pairwise_data = np.vstack([data_group_a, data_group_b])
            pairwise_communities = np.append(communities_group_a, communities_group_b)
            projected_points = _lda_based_projection(pairwise_data, pairwise_communities)
            if not projected_points.size == 0:
                scores = _convert_points_to_one_dimension(projected_points)
        elif variant == SeparabilityVariant.TSPS:
            # load default settings
            if concorde_settings is None:
                concorde_settings = ConcordeSettings()  # default

            pairwise_data = np.vstack([data_group_a, data_group_b])
            metadata[index_group_combination]["pairwise_data"] = pairwise_data

            pairwise_communities = np.append(communities_group_a, communities_group_b)
            metadata[index_group_combination]["pairwise_communities"] = pairwise_communities

            best_tour = _tsp_based_projection(pairwise_data, concorde_settings)
            metadata[index_group_combination]["best_tour"] = best_tour

            if not best_tour.size == 0:
                separability_path = _convert_tour_to_one_dimension(best_tour, pairwise_data, pairwise_communities)
                scores, source_nodes, target_nodes, edge_weights, cut_start_node, cut_end_node = separability_path

                metadata[index_group_combination]["source_nodes"] = source_nodes
                metadata[index_group_combination]["target_nodes"] = target_nodes
                metadata[index_group_combination]["edge_weights"] = edge_weights
                metadata[index_group_combination]["cut_start_node"] = cut_start_node
                metadata[index_group_combination]["cut_end_node"] = cut_end_node
        else:
            raise RuntimeError('invalid community separability variant')

        metadata[index_group_combination]["scores"] = scores

        if scores is None:
            auc_values = np.append(auc_values, 0)
            aupr_values = np.append(auc_values, 0)
            mcc_values = np.append(auc_values, 0)
            continue

        # construct community membership
        communities_membership = np.concatenate((communities_group_a, communities_group_b), axis=0)
        current_positive_community_class = _extract_positive_class(communities_membership, positives)

        auc, aupr = _compute_auc_aupr(communities_membership, scores, current_positive_community_class)
        auc_values = np.append(auc_values, auc)
        aupr_values = np.append(aupr_values, aupr)

        mcc = _compute_mcc(communities_membership, scores, current_positive_community_class)
        mcc_values = np.append(mcc_values, mcc)

    # compile all values from the different pairwise community combinations
    delta_degrees_of_freedom = 0
    if total_communities > 2:
        delta_degrees_of_freedom = 1

    # correct values (apply custom penalization)
    corrected_auc = np.mean(auc_values) / (np.std(auc_values, ddof=delta_degrees_of_freedom) + 1)
    corrected_aupr = np.mean(aupr_values) / (np.std(aupr_values, ddof=delta_degrees_of_freedom) + 1)
    corrected_mcc = np.mean(mcc_values) / (np.std(mcc_values, ddof=delta_degrees_of_freedom) + 1)

    measures = dict(
        auc=corrected_auc,
        aupr=corrected_aupr,
        mcc=corrected_mcc
    )

    if permutations is not None:
        measures = _compute_permutations(measures, metadata, communities, positives, permutations)

    return measures, metadata
