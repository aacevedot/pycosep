import unittest

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant
from tests.test_data import _half_kernel, _parallel_lines, _circles, _rhombus, _spirals


class TestLDPSCommunitySeparability(unittest.TestCase):
    def test_ldps_returns_expected_indices_when_half_kernel_data_without_permutations(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.7067, round(indices['auc'], 4))
        self.assertEqual(0.5421, round(indices['aupr'], 4))
        self.assertEqual(0.1833, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_half_kernel_data_with_1000_permutations(self):
        embedding, communities = _half_kernel()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.7067, round(auc_results['original_value'], 4))
        self.assertEqual(0.0300, round(auc_results['p_value'], 4))  # MATLAB: 0.0330
        self.assertEqual(0.5781, round(auc_results['mean'], 4))  # MATLAB: 0.5785
        self.assertEqual(0.7800, round(auc_results['max'], 4))  # MATLAB: 0.8033
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0570, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0587
        self.assertEqual(0.0018, round(auc_results['standard_error'], 4))  # MATLAB: 0.0019

        aupr_results = permutations['aupr']
        self.assertEqual(0.5421, round(aupr_results['original_value'], 4))
        self.assertEqual(0.3077, round(aupr_results['p_value'], 4))  # MATLAB: 0.3397
        self.assertEqual(0.5119, round(aupr_results['mean'], 4))  # MATLAB: 0.5152
        self.assertEqual(0.7811, round(aupr_results['max'], 4))  # MATLAB: 0.8251
        self.assertEqual(0.3867, round(aupr_results['min'], 4))  # MATLAB: 0.3882
        self.assertEqual(0.0726, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0753
        self.assertEqual(0.0023, round(aupr_results['standard_error'], 4))  # MATLAB: 0.0024

        mcc_results = permutations['mcc']
        self.assertEqual(0.1833, round(mcc_results['original_value'], 4))
        self.assertEqual(0.4476, round(mcc_results['p_value'], 4))  # MATLAB: 0.4535
        self.assertEqual(0.1262, round(mcc_results['mean'], 4))  # MATLAB: 0.1269
        self.assertEqual(0.5333, round(mcc_results['max'], 4))  # MATLAB: 0.6500
        self.assertEqual(-0.1667, round(mcc_results['min'], 4))
        self.assertEqual(0.1145, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1153
        self.assertEqual(0.0036, round(mcc_results['standard_error'], 4))

    def test_ldps_returns_expected_indices_when_circles_data_without_permutations(self):
        embedding, communities = _circles()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.5100, round(indices['auc'], 4))
        self.assertEqual(0.6425, round(indices['aupr'], 4))
        self.assertEqual(0.0000, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_circles_data_with_1000_permutations(self):
        embedding, communities = _circles()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.5100, round(auc_results['original_value'], 4))
        self.assertEqual(0.9251, round(auc_results['p_value'], 4))  # MATLAB: 0.9131
        self.assertEqual(0.5730, round(auc_results['mean'], 4))  # MATLAB: 0.5712
        self.assertEqual(0.8062, round(auc_results['max'], 4))  # MATLAB: 0.8325
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0548, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0526
        self.assertEqual(0.0017, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6425, round(aupr_results['original_value'], 4))
        self.assertEqual(0.1578, round(aupr_results['p_value'], 4))  # MATLAB: 0.1249
        self.assertEqual(0.5788, round(aupr_results['mean'], 4))  # MATLAB: 0.5692
        self.assertEqual(0.8237, round(aupr_results['max'], 4))  # MATLAB: 0.8109
        self.assertEqual(0.4577, round(aupr_results['min'], 4))  # MATLAB: 0.4613
        self.assertEqual(0.0625, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0020, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(1.0000, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1222, round(mcc_results['mean'], 4))  # MATLAB: 0.1195
        self.assertEqual(0.5000, round(mcc_results['max'], 4))  # MATLAB: 0.6000
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.1003, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.0948
        self.assertEqual(0.0032, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0030

    def test_ldps_returns_expected_indices_when_rhombus_data_without_permutations(self):
        embedding, communities = _rhombus()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_rhombus_data_with_1000_permutations(self):
        embedding, communities = _rhombus()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS,
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

    def test_ldps_returns_expected_indices_when_spirals_data_without_permutations(self):
        embedding, communities = _spirals()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.6128, round(indices['auc'], 4))
        self.assertEqual(0.6192, round(indices['aupr'], 4))
        self.assertEqual(0.1780, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_spirals_data_with_1000_permutations(self):
        embedding, communities = _spirals()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.6128, round(auc_results['original_value'], 4))
        self.assertEqual(0.1738, round(auc_results['p_value'], 4))
        self.assertEqual(0.5642, round(auc_results['mean'], 4))  # MATLAB: 0.5645
        self.assertEqual(0.7446, round(auc_results['max'], 4))  # MATLAB: 0.7745
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0486, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0499
        self.assertEqual(0.0015, round(auc_results['standard_error'], 4))  # MATLAB: 0.0016

        aupr_results = permutations['aupr']
        self.assertEqual(0.6192, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0290, round(aupr_results['p_value'], 4))  # MATLAB: 0.0370
        self.assertEqual(0.4848, round(aupr_results['mean'], 4))  # MATLAB: 0.4868
        self.assertEqual(0.7352, round(aupr_results['max'], 4))  # MATLAB: 0.7580
        self.assertEqual(0.3812, round(aupr_results['min'], 4))  # MATLAB: 0.3797
        self.assertEqual(0.0606, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0627
        self.assertEqual(0.0019, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.1780, round(mcc_results['original_value'], 4))
        self.assertEqual(0.3207, round(mcc_results['p_value'], 4))  # MATLAB: 0.3117
        self.assertEqual(0.1066, round(mcc_results['mean'], 4))  # MATLAB: 0.1033
        self.assertEqual(0.4022, round(mcc_results['max'], 4))
        self.assertEqual(-0.1209, round(mcc_results['min'], 4))
        self.assertEqual(0.0948, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.0923
        self.assertEqual(0.0030, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0029

    def test_ldps_returns_expected_indices_when_parallel_lines_data_without_permutations(self):
        embedding, communities = _parallel_lines()

        indices, metadata = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.5833, round(indices['auc'], 4))  # MATLAB: 1.0000
        self.assertEqual(0.6199, round(indices['aupr'], 4))  # MATLAB: 1.0000
        self.assertEqual(0.0000, round(indices['mcc'], 4))  # MATLAB: 1.0000

    def test_ldps_returns_expected_indices_when_parallel_lines_data_with_1000_permutations(self):
        embedding, communities = _parallel_lines()

        permutations, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS,
            permutations=1000)

        auc_results = permutations['auc']
        self.assertEqual(0.5833, round(auc_results['original_value'], 4))  # MATLAB: 1.0000
        self.assertEqual(0.6973, round(auc_results['p_value'], 4))  # MATLAB: 0.0020
        self.assertEqual(0.6398, round(auc_results['mean'], 4))  # MATLAB: 0.6349
        self.assertEqual(1.0000, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.1022, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0981
        self.assertEqual(0.0032, round(auc_results['standard_error'], 4))  # MATLAB: 0.0031

        aupr_results = permutations['aupr']
        self.assertEqual(0.6199, round(aupr_results['original_value'], 4))  # MATLAB: 1.0000
        self.assertEqual(0.5934, round(aupr_results['p_value'], 4))  # MATLAB: 0.0020
        self.assertEqual(0.6555, round(aupr_results['mean'], 4))  # MATLAB: 0.6504
        self.assertEqual(1.0000, round(aupr_results['max'], 4))
        self.assertEqual(0.4444, round(aupr_results['min'], 4))  # MATLAB: 0.4422
        self.assertEqual(0.1223, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.1193
        self.assertEqual(0.0039, round(aupr_results['standard_error'], 4))  # MATLAB: 0.0038

        mcc_results = permutations['mcc']
        self.assertEqual(0.0000, round(mcc_results['original_value'], 4))  # MATLAB: 1.0000
        self.assertEqual(1.0000, round(mcc_results['p_value'], 4))  # MATLAB: 0.0020
        self.assertEqual(0.2127, round(mcc_results['mean'], 4))  # MATLAB: 0.2107
        self.assertEqual(1.0000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.2079, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.2052
        self.assertEqual(0.0066, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0065
