import unittest

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant
from tests.test_data import _half_kernel, _parallel_lines, _circles, _rhombus, _spirals


class TestTSPSCommunitySeparability(unittest.TestCase):
    def test_tsps_returns_expected_indices_when_half_kernel_data_without_permutations(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_half_kernel_data_with_1000_permutations(self):
        embedding, communities = _half_kernel()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(1.0000, round(auc_results['original_value'], 4))
        self.assertEqual(0.0010, round(auc_results['p_value'], 4))
        self.assertEqual(0.5813, round(auc_results['mean'], 4))
        self.assertEqual(0.8300, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0599, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0019, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(1.0000, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0010, round(aupr_results['p_value'], 4))
        self.assertEqual(0.5160, round(aupr_results['mean'], 4))
        self.assertEqual(0.8032, round(aupr_results['max'], 4))
        self.assertEqual(0.3884, round(aupr_results['min'], 4))
        self.assertEqual(0.0760, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0759
        self.assertEqual(0.0024, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(1.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(0.0010, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1283, round(mcc_results['mean'], 4))
        self.assertEqual(0.6500, round(mcc_results['max'], 4))
        self.assertEqual(-0.1667, round(mcc_results['min'], 4))
        self.assertEqual(0.1201, round(mcc_results['standard_deviation'], 4))
        self.assertEqual(0.0038, round(mcc_results['standard_error'], 4))

    def test_tsps_returns_expected_indices_when_circles_data_without_permutations(self):
        embedding, communities = _circles()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_circles_data_with_1000_permutations(self):
        embedding, communities = _circles()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(1.0000, round(auc_results['original_value'], 4))
        self.assertEqual(0.0010, round(auc_results['p_value'], 4))
        self.assertEqual(0.5781, round(auc_results['mean'], 4))
        self.assertEqual(0.7850, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0569, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0018, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(1.0000, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0010, round(aupr_results['p_value'], 4))
        self.assertEqual(0.5824, round(aupr_results['mean'], 4))  # MATLAB: 0.5822
        self.assertEqual(0.8092, round(aupr_results['max'], 4))
        self.assertEqual(0.4563, round(aupr_results['min'], 4))  # MATLAB: 0.4557
        self.assertEqual(0.0677, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0679
        self.assertEqual(0.0021, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(1.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(0.0010, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1284, round(mcc_results['mean'], 4))
        self.assertEqual(0.5000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.1056, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1057
        self.assertEqual(0.0033, round(mcc_results['standard_error'], 4))

    def test_tsps_returns_expected_indices_when_rhombus_data_without_permutations(self):
        embedding, communities = _rhombus()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(0.9100, round(indices['auc'], 4))
        self.assertEqual(0.9212, round(indices['aupr'], 4))
        self.assertEqual(0.4000, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_rhombus_data_with_1000_permutations(self):
        embedding, communities = _rhombus()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.9100, round(auc_results['original_value'], 4))
        self.assertEqual(0.0030, round(auc_results['p_value'], 4))
        self.assertEqual(0.6083, round(auc_results['mean'], 4))
        self.assertEqual(0.9500, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0801, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0025, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.9212, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0020, round(aupr_results['p_value'], 4))
        self.assertEqual(0.6150, round(aupr_results['mean'], 4))  # MATLAB: 0.6149
        self.assertEqual(0.9569, round(aupr_results['max'], 4))
        self.assertEqual(0.4505, round(aupr_results['min'], 4))
        self.assertEqual(0.0948, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0949
        self.assertEqual(0.0030, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.4000, round(mcc_results['original_value'], 4))
        self.assertEqual(0.2008, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1830, round(mcc_results['mean'], 4))
        self.assertEqual(0.8000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.1533, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1534
        self.assertEqual(0.0048, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0049

    def test_tsps_returns_expected_indices_when_spirals_data_without_permutations(self):
        embedding, communities = _spirals()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(0.8614, round(indices['auc'], 4))
        self.assertEqual(0.8446, round(indices['aupr'], 4))
        self.assertEqual(0.5516, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_spirals_data_with_1000_permutations(self):
        embedding, communities = _spirals()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.8614, round(auc_results['original_value'], 4))
        self.assertEqual(0.0010, round(auc_results['p_value'], 4))
        self.assertEqual(0.5603, round(auc_results['mean'], 4))
        self.assertEqual(0.7391, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0453, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0014, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.8446, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0010, round(aupr_results['p_value'], 4))
        self.assertEqual(0.4844, round(aupr_results['mean'], 4))
        self.assertEqual(0.7469, round(aupr_results['max'], 4))
        self.assertEqual(0.3863, round(aupr_results['min'], 4))
        self.assertEqual(0.0589, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0590
        self.assertEqual(0.0019, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.5516, round(mcc_results['original_value'], 4))
        self.assertEqual(0.0010, round(mcc_results['p_value'], 4))
        self.assertEqual(0.0948, round(mcc_results['mean'], 4))
        self.assertEqual(0.4769, round(mcc_results['max'], 4))
        self.assertEqual(-0.1209, round(mcc_results['min'], 4))
        self.assertEqual(0.0877, round(mcc_results['standard_deviation'], 4))
        self.assertEqual(0.0028, round(mcc_results['standard_error'], 4))

    def test_tsps_returns_expected_indices_when_parallel_lines_data_without_permutations(self):
        embedding, communities = _parallel_lines()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_parallel_lines_data_with_1000_permutations(self):
        embedding, communities = _parallel_lines()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(1.0000, round(auc_results['original_value'], 4))
        self.assertEqual(0.0020, round(auc_results['p_value'], 4))
        self.assertEqual(0.6358, round(auc_results['mean'], 4))
        self.assertEqual(1.0000, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.1005, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.1006
        self.assertEqual(0.0032, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(1.0000, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0020, round(aupr_results['p_value'], 4))
        self.assertEqual(0.6501, round(aupr_results['mean'], 4))  # MATLAB: 0.6503
        self.assertEqual(1.0000, round(aupr_results['max'], 4))
        self.assertEqual(0.4422, round(aupr_results['min'], 4))
        self.assertEqual(0.1204, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.1203
        self.assertEqual(0.0038, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(1.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(0.0020, round(mcc_results['p_value'], 4))
        self.assertEqual(0.2107, round(mcc_results['mean'], 4))
        self.assertEqual(1.0000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.2051, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.2052
        self.assertEqual(0.0065, round(mcc_results['standard_error'], 4))
