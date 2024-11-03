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
        self.assertEqual(0.0440, round(auc_results['p_value'], 4))  # MATLAB: 0.0480
        self.assertEqual(0.5781, round(auc_results['mean'], 4))  # MATLAB: 0.5782
        self.assertEqual(0.7967, round(auc_results['max'], 4))  # MATLAB: 0.8067
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0570, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0582
        self.assertEqual(0.0018, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.5228, round(aupr_results['original_value'], 4))
        self.assertEqual(0.4036, round(aupr_results['p_value'], 4))  # MATLAB: 0.4046
        self.assertEqual(0.5134, round(aupr_results['mean'], 4))  # MATLAB: 0.5150
        self.assertEqual(0.7628, round(aupr_results['max'], 4))  # MATLAB: 0.8340
        self.assertEqual(0.3907, round(aupr_results['min'], 4))  # MATLAB: 0.3871
        self.assertEqual(0.0728, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0753
        self.assertEqual(0.0023, round(aupr_results['standard_error'], 4))  # MATLAB: 0.0024

        mcc_results = permutations['mcc']
        self.assertEqual(0.1833, round(mcc_results['original_value'], 4))
        self.assertEqual(0.4356, round(mcc_results['p_value'], 4))  # MATLAB: 0.4605
        self.assertEqual(0.1235, round(mcc_results['mean'], 4))  # MATLAB: 0.1283
        self.assertEqual(0.6500, round(mcc_results['max'], 4))
        self.assertEqual(-0.1667, round(mcc_results['min'], 4))
        self.assertEqual(0.1164, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1131
        self.assertEqual(0.0037, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0036

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
        self.assertEqual(0.9241, round(auc_results['p_value'], 4))  # MATLAB: 0.9091
        self.assertEqual(0.5728, round(auc_results['mean'], 4))  # MATLAB: 0.5712
        self.assertEqual(0.8038, round(auc_results['max'], 4))  # MATLAB: 0.8325
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0547, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0527
        self.assertEqual(0.0017, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6425, round(aupr_results['original_value'], 4))
        self.assertEqual(0.1558, round(aupr_results['p_value'], 4))  # MATLAB: 0.1259
        self.assertEqual(0.5785, round(aupr_results['mean'], 4))  # MATLAB: 0.5690
        self.assertEqual(0.8198, round(aupr_results['max'], 4))  # MATLAB: 0.8109
        self.assertEqual(0.4580, round(aupr_results['min'], 4))  # MATLAB: 0.4603
        self.assertEqual(0.0625, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0626
        self.assertEqual(0.0020, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(1.0000, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1222, round(mcc_results['mean'], 4))  # MATLAB: 0.1195
        self.assertEqual(0.5000, round(mcc_results['max'], 4))  # MATLAB: 0.6000
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.1003, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.0948
        self.assertEqual(0.0032, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0030

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
        self.assertEqual(0.6056, round(auc_results['mean'], 4))  # MATLAB: 0.6047
        self.assertEqual(0.8850, round(auc_results['max'], 4))  # MATLAB: 0.9250
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0750, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0796
        self.assertEqual(0.0024, round(auc_results['standard_error'], 4))  # MATLAB: 0.0025

        aupr_results = permutations['aupr']
        self.assertEqual(1.0000, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0010, round(aupr_results['p_value'], 4))
        self.assertEqual(0.6098, round(aupr_results['mean'], 4))  # MATLAB: 0.6120
        self.assertEqual(0.9137, round(aupr_results['max'], 4))
        self.assertEqual(0.4401, round(aupr_results['min'], 4))  # MATLAB: 0.4350
        self.assertEqual(0.0977, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.1008
        self.assertEqual(0.0031, round(aupr_results['standard_error'], 4))  # MATLAB: 0.0032

        mcc_results = permutations['mcc']
        self.assertEqual(1.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(0.0010, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1688, round(mcc_results['mean'], 4))  # MATLAB: 0.1646
        self.assertEqual(0.6000, round(mcc_results['max'], 4))  # MATLAB: 0.8000
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.1512, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1526
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
        self.assertEqual(0.1618, round(auc_results['p_value'], 4))  # MATLAB: 0.1708
        self.assertEqual(0.5635, round(auc_results['mean'], 4))  # MATLAB: 0.5646
        self.assertEqual(0.7337, round(auc_results['max'], 4))  # MATLAB: 0.7690
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0471, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0487
        self.assertEqual(0.0015, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6132, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0370, round(aupr_results['p_value'], 4))  # MATLAB: 0.0450
        self.assertEqual(0.4844, round(aupr_results['mean'], 4))  # MATLAB: 0.4872
        self.assertEqual(0.7084, round(aupr_results['max'], 4))  # MATLAB: 0.7333
        self.assertEqual(0.3783, round(aupr_results['min'], 4))  # MATLAB: 0.3801
        self.assertEqual(0.0602, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0621
        self.assertEqual(0.0019, round(aupr_results['standard_error'], 4))  # MATLAB: 0.0020

        mcc_results = permutations['mcc']
        self.assertEqual(0.2527, round(mcc_results['original_value'], 4))
        self.assertEqual(0.1149, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1023, round(mcc_results['mean'], 4))  # MATLAB: 0.1034
        self.assertEqual(0.4022, round(mcc_results['max'], 4))  # MATLAB: 0.4769
        self.assertEqual(-0.1209, round(mcc_results['min'], 4))
        self.assertEqual(0.0917, round(mcc_results['standard_deviation'], 4))
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
        self.assertEqual(0.4116, round(auc_results['p_value'], 4))  # MATLAB: 0.4276
        self.assertEqual(0.6381, round(auc_results['mean'], 4))  # MATLAB: 0.6314
        self.assertEqual(0.9861, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.1012, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0999
        self.assertEqual(0.0032, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6944, round(aupr_results['original_value'], 4))
        self.assertEqual(0.4046, round(aupr_results['p_value'], 4))  # MATLAB: 0.3776
        self.assertEqual(0.6486, round(aupr_results['mean'], 4))  # MATLAB: 0.6420
        self.assertEqual(0.9881, round(aupr_results['max'], 4))
        self.assertEqual(0.4349, round(aupr_results['min'], 4))  # MATLAB: 0.4359
        self.assertEqual(0.1303, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.1290
        self.assertEqual(0.0041, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.3333, round(mcc_results['original_value'], 4))
        self.assertEqual(0.5664, round(mcc_results['p_value'], 4))  # MATLAB: 0.5724
        self.assertEqual(0.2160, round(mcc_results['mean'], 4))  # MATLAB: 0.2173
        self.assertEqual(1.0000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.2103, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.2090
        self.assertEqual(0.0067, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0066
