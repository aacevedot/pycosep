import unittest

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant
from tests.test_data import _half_kernel, _parallel_lines, _circles, _rhombus, _spirals


class TestCPSCommunitySeparability(unittest.TestCase):
    def test_cps_returns_expected_indices_when_half_kernel_data_without_permutations(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6933, round(indices['auc'], 4))
        self.assertEqual(0.5228, round(indices['aupr'], 4))
        self.assertEqual(0.1833, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_half_kernel_data_with_1000_permutations(self):
        embedding, communities = _half_kernel()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.6933, round(auc_results['original_value'], 4))
        self.assertEqual(0.0490, round(auc_results['p_value'], 4))  # MATLAB: 0.0480
        self.assertEqual(0.5782, round(auc_results['mean'], 4))
        self.assertEqual(0.8067, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0582, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0018, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.5228, round(aupr_results['original_value'], 4))
        self.assertEqual(0.4046, round(aupr_results['p_value'], 4))
        self.assertEqual(0.5149, round(aupr_results['mean'], 4))  # MATLAB: 0.5150
        self.assertEqual(0.8340, round(aupr_results['max'], 4))
        self.assertEqual(0.3871, round(aupr_results['min'], 4))
        self.assertEqual(0.0753, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0024, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.1833, round(mcc_results['original_value'], 4))
        self.assertEqual(0.4605, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1283, round(mcc_results['mean'], 4))
        self.assertEqual(0.6500, round(mcc_results['max'], 4))
        self.assertEqual(-0.1667, round(mcc_results['min'], 4))
        self.assertEqual(0.1131, round(mcc_results['standard_deviation'], 4))
        self.assertEqual(0.0036, round(mcc_results['standard_error'], 4))

    def test_cps_returns_expected_indices_when_circles_data_without_permutations(self):
        embedding, communities = _circles()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.5100, round(indices['auc'], 4))
        self.assertEqual(0.6425, round(indices['aupr'], 4))
        self.assertEqual(0.0000, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_circles_data_with_1000_permutations(self):
        embedding, communities = _circles()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.5100, round(auc_results['original_value'], 4))
        self.assertEqual(0.9091, round(auc_results['p_value'], 4))
        self.assertEqual(0.5712, round(auc_results['mean'], 4))
        self.assertEqual(0.8325, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0527, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0017, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6425, round(aupr_results['original_value'], 4))
        self.assertEqual(0.1259, round(aupr_results['p_value'], 4))
        self.assertEqual(0.5690, round(aupr_results['mean'], 4))
        self.assertEqual(0.8109, round(aupr_results['max'], 4))
        self.assertEqual(0.4603, round(aupr_results['min'], 4))
        self.assertEqual(0.0626, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0020, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(1.0000, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1195, round(mcc_results['mean'], 4))
        self.assertEqual(0.6000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.0947, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.0948
        self.assertEqual(0.0030, round(mcc_results['standard_error'], 4))

    def test_cps_returns_expected_indices_when_rhombus_data_without_permutations(self):
        embedding, communities = _rhombus()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_rhombus_data_with_1000_permutations(self):
        embedding, communities = _rhombus()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(1.0000, round(auc_results['original_value'], 4))
        self.assertEqual(0.0010, round(auc_results['p_value'], 4))
        self.assertEqual(0.6048, round(auc_results['mean'], 4))  # MATLAB: 0.6047
        self.assertEqual(0.9250, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0796, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0025, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(1.0000, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0010, round(aupr_results['p_value'], 4))
        self.assertEqual(0.6120, round(aupr_results['mean'], 4))
        self.assertEqual(0.9137, round(aupr_results['max'], 4))
        self.assertEqual(0.4350, round(aupr_results['min'], 4))
        self.assertEqual(0.1008, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0032, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(1.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(0.0010, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1646, round(mcc_results['mean'], 4))
        self.assertEqual(0.8000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.1525, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1526
        self.assertEqual(0.0048, round(mcc_results['standard_error'], 4))

    def test_cps_returns_expected_indices_when_spirals_data_without_permutations(self):
        embedding, communities = _spirals()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6128, round(indices['auc'], 4))
        self.assertEqual(0.6132, round(indices['aupr'], 4))
        self.assertEqual(0.2527, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_spirals_data_with_1000_permutations(self):
        embedding, communities = _spirals()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.6128, round(auc_results['original_value'], 4))
        self.assertEqual(0.1708, round(auc_results['p_value'], 4))
        self.assertEqual(0.5646, round(auc_results['mean'], 4))
        self.assertEqual(0.7690, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0487, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0015, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6132, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0450, round(aupr_results['p_value'], 4))
        self.assertEqual(0.4873, round(aupr_results['mean'], 4))  # MATLAB: 0.4872
        self.assertEqual(0.7333, round(aupr_results['max'], 4))
        self.assertEqual(0.3801, round(aupr_results['min'], 4))
        self.assertEqual(0.0620, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0621
        self.assertEqual(0.0020, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.2527, round(mcc_results['original_value'], 4))
        self.assertEqual(0.1149, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1034, round(mcc_results['mean'], 4))
        self.assertEqual(0.4769, round(mcc_results['max'], 4))
        self.assertEqual(-0.1209, round(mcc_results['min'], 4))
        self.assertEqual(0.0916, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.0917
        self.assertEqual(0.0029, round(mcc_results['standard_error'], 4))

    def test_cps_returns_expected_indices_when_parallel_lines_data_without_permutations(self):
        embedding, communities = _parallel_lines()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6528, round(indices['auc'], 4))
        self.assertEqual(0.6944, round(indices['aupr'], 4))
        self.assertEqual(0.3333, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_parallel_lines_data_with_1000_permutations(self):
        embedding, communities = _parallel_lines()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.6528, round(auc_results['original_value'], 4))
        self.assertEqual(0.3906, round(auc_results['p_value'], 4))  # MATLAB: 0.4276
        self.assertEqual(0.6314, round(auc_results['mean'], 4))
        self.assertEqual(0.9861, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0999, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0032, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6944, round(aupr_results['original_value'], 4))
        self.assertEqual(0.3776, round(aupr_results['p_value'], 4))
        self.assertEqual(0.6420, round(aupr_results['mean'], 4))
        self.assertEqual(0.9881, round(aupr_results['max'], 4))
        self.assertEqual(0.4359, round(aupr_results['min'], 4))
        self.assertEqual(0.1290, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0041, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.3333, round(mcc_results['original_value'], 4))
        self.assertEqual(0.5724, round(mcc_results['p_value'], 4))
        self.assertEqual(0.2173, round(mcc_results['mean'], 4))
        self.assertEqual(1.0000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.2089, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.2090
        self.assertEqual(0.0066, round(mcc_results['standard_error'], 4))
