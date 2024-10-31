import itertools
import os
import subprocess
import time
import uuid
import warnings

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from pycosep.runtime_settings import RuntimeSettings
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


def _tsp_based_projection(pairwise_data, runtime_settings):
    # check if 'Concorde' path is specified in runtime settings
    if runtime_settings.concorde_path == '':
        raise ValueError("'Concorde' path not specified in runtime settings")

    # sanity check to avoid overriding TSP input and output files
    while True:
        short_uuid = str(uuid.uuid4())[-12:].replace('-', '')

        file_name = short_uuid
        file_path = os.path.join(runtime_settings.temp_path, file_name)

        file_tsp_name = file_name + '.tsp'
        file_tsp_path = os.path.join(runtime_settings.temp_path, file_tsp_name)
        file_sol_name = file_name + '.sol'
        file_sol_path = os.path.join(runtime_settings.temp_path, file_sol_name)

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
        file.write(f"NAME : TSPS Concorde\n")
        file.write(f"COMMENT : Scaling factor {scaling_factor}\n")
        file.write(f"TYPE : TSP\n")
        file.write(f"DIMENSION : {total_nodes}\n")
        file.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write(f"NODE_COORD_SECTION\n")
        for ix in range(total_nodes):
            file.write(f"{ix + 1} {scaled_embedding[ix, 0]} {scaled_embedding[ix, 1]}\n")
        file.write(f"EOF\n")

    # execute Concorde
    command = f"{runtime_settings.concorde_path} -s 40 -x -o {file_sol_name} {file_tsp_name}"

    result = subprocess.run(command, cwd=runtime_settings.temp_path, capture_output=True, text=True, check=False)
    if result.returncode != 0 and result.returncode != 255:
        raise RuntimeError(f"Error executing Concorde command: {result.stdout}")

    # read Concorde solution file
    try:
        with open(file_sol_path, 'r') as file:
            # skip the first element (number of nodes)
            loaded_tour = np.genfromtxt(file, skip_header=1, dtype=int, filling_values=np.nan)
    except Exception as e:
        warnings.warn(f"Cannot process Concorde solution: {e}", RuntimeWarning)
        return np.array([])

    # Concorde TSP tour port-processing
    best_route = loaded_tour[~np.isnan(loaded_tour)].astype(int)

    # clean up TSP files
    if os.path.isfile(file_tsp_path):
        os.remove(file_tsp_path)
    if os.path.isfile(file_sol_path):
        os.remove(file_sol_path)

    # TODO: Probably, not needed
    # temp_file_sol = os.path.join(runtime_settings.root_path, file_name + '.sol')
    # if os.path.isfile(temp_file_sol):
    #    os.remove(temp_file_sol)

    return best_route


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


# TODO: Set runtime settings correctly (e.g., default value)
def compute_separability(embedding, communities, positives=None, variant=SeparabilityVariant.CPS, permutations=None,
                         runtime_settings=None):
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

    # load default settings
    if runtime_settings is None:
        runtime_settings = RuntimeSettings()

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
        metadata[index_group_combination]["data_group_a"] = communities_group_a

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
            pairwise_data = np.vstack([data_group_a, data_group_b])
            metadata[index_group_combination]["pairwise_data"] = pairwise_data

            pairwise_communities = np.append(communities_group_a, communities_group_b)
            metadata[index_group_combination]["pairwise_communities"] = pairwise_communities

            best_tour = _tsp_based_projection(pairwise_data, runtime_settings)
            metadata[index_group_combination]["best_tour"] = best_tour

            if not best_tour.size == 0:
                scores, source_nodes, target_nodes, edge_weights, cut_start_node, cut_end_node = _convert_tour_to_one_dimension(
                    best_tour, pairwise_data, pairwise_communities)
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

        current_positive_community_class = None
        for o in range(len(positives)):
            if np.any(communities_membership == positives[o]):
                current_positive_community_class = positives[o]
                break

        if current_positive_community_class is None:
            raise RuntimeError('impossible to set the current positive community class')

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
        raise NotImplementedError("'permutations' handling not implemented yet")

    return measures, metadata
