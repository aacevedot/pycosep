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
        self.assertEqual(0.0330, round(auc_results['p_value'], 4))
        self.assertEqual(0.5785, round(auc_results['mean'], 4))
        self.assertEqual(0.8033, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0586, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0587
        self.assertEqual(0.0019, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.5421, round(aupr_results['original_value'], 4))
        self.assertEqual(0.3397, round(aupr_results['p_value'], 4))
        self.assertEqual(0.5153, round(aupr_results['mean'], 4))  # MATLAB: 0.5152
        self.assertEqual(0.8251, round(aupr_results['max'], 4))
        self.assertEqual(0.3882, round(aupr_results['min'], 4))
        self.assertEqual(0.0752, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.0753
        self.assertEqual(0.0024, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.1833, round(mcc_results['original_value'], 4))
        self.assertEqual(0.4535, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1269, round(mcc_results['mean'], 4))
        self.assertEqual(0.6500, round(mcc_results['max'], 4))
        self.assertEqual(-0.1667, round(mcc_results['min'], 4))
        self.assertEqual(0.1152, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.1153
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
        self.assertEqual(0.9131, round(auc_results['p_value'], 4))
        self.assertEqual(0.5712, round(auc_results['mean'], 4))
        self.assertEqual(0.8325, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0526, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0017, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6425, round(aupr_results['original_value'], 4))
        self.assertEqual(0.1249, round(aupr_results['p_value'], 4))
        self.assertEqual(0.5692, round(aupr_results['mean'], 4))
        self.assertEqual(0.8109, round(aupr_results['max'], 4))
        self.assertEqual(0.4613, round(aupr_results['min'], 4))
        self.assertEqual(0.0625, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0020, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.0000, round(mcc_results['original_value'], 4))
        self.assertEqual(1.0000, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1195, round(mcc_results['mean'], 4))
        self.assertEqual(0.6000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.0947, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.0948
        self.assertEqual(0.0030, round(mcc_results['standard_error'], 4))

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
        self.assertEqual(0.1748, round(auc_results['p_value'], 4))  # MATLAB: 0.1738
        self.assertEqual(0.5645, round(auc_results['mean'], 4))
        self.assertEqual(0.7745, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.0499, round(auc_results['standard_deviation'], 4))
        self.assertEqual(0.0016, round(auc_results['standard_error'], 4))

        aupr_results = permutations['aupr']
        self.assertEqual(0.6192, round(aupr_results['original_value'], 4))
        self.assertEqual(0.0370, round(aupr_results['p_value'], 4))
        self.assertEqual(0.4867, round(aupr_results['mean'], 4))  # MATLAB: 0.4868
        self.assertEqual(0.7580, round(aupr_results['max'], 4))
        self.assertEqual(0.3797, round(aupr_results['min'], 4))
        self.assertEqual(0.0627, round(aupr_results['standard_deviation'], 4))
        self.assertEqual(0.0020, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.1780, round(mcc_results['original_value'], 4))
        self.assertEqual(0.3117, round(mcc_results['p_value'], 4))
        self.assertEqual(0.1033, round(mcc_results['mean'], 4))
        self.assertEqual(0.4022, round(mcc_results['max'], 4))
        self.assertEqual(-0.1209, round(mcc_results['min'], 4))
        self.assertEqual(0.0923, round(mcc_results['standard_deviation'], 4))
        self.assertEqual(0.0029, round(mcc_results['standard_error'], 4))

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
        self.assertEqual(0.6863, round(auc_results['p_value'], 4))  # MATLAB: 0.0020
        self.assertEqual(0.6336, round(auc_results['mean'], 4))  # MATLAB: 0.6349
        self.assertEqual(1.0000, round(auc_results['max'], 4))
        self.assertEqual(0.5000, round(auc_results['min'], 4))
        self.assertEqual(0.1000, round(auc_results['standard_deviation'], 4))  # MATLAB: 0.0981
        self.assertEqual(0.0032, round(auc_results['standard_error'], 4))  # MATLAB: 0.0031

        aupr_results = permutations['aupr']
        self.assertEqual(0.6199, round(aupr_results['original_value'], 4))  # MATLAB: 1.0000
        self.assertEqual(0.5694, round(aupr_results['p_value'], 4))  # MATLAB: 0.0020
        self.assertEqual(0.6494, round(aupr_results['mean'], 4))  # MATLAB: 0.6504
        self.assertEqual(1.0000, round(aupr_results['max'], 4))
        self.assertEqual(0.4444, round(aupr_results['min'], 4))  # MATLAB: 0.4422
        self.assertEqual(0.1207, round(aupr_results['standard_deviation'], 4))  # MATLAB: 0.1193
        self.assertEqual(0.0038, round(aupr_results['standard_error'], 4))

        mcc_results = permutations['mcc']
        self.assertEqual(0.0000, round(mcc_results['original_value'], 4))  # MATLAB: 1.0000
        self.assertEqual(1.0000, round(mcc_results['p_value'], 4))  # MATLAB: 0.0020
        self.assertEqual(0.2050, round(mcc_results['mean'], 4))  # MATLAB: 0.2107
        self.assertEqual(1.0000, round(mcc_results['max'], 4))
        self.assertEqual(0.0000, round(mcc_results['min'], 4))
        self.assertEqual(0.2073, round(mcc_results['standard_deviation'], 4))  # MATLAB: 0.2052
        self.assertEqual(0.0066, round(mcc_results['standard_error'], 4))  # MATLAB: 0.0065
