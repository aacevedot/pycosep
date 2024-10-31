import unittest

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant
from tests.test_data import _half_kernel


class TestCommunitySeparability(unittest.TestCase):
    def test_tsps_returns_expected_indices_when_half_kernel_data(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_half_kernel_data(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.7067, round(indices['auc'], 4))
        self.assertEqual(0.5421, round(indices['aupr'], 4))
        self.assertEqual(0.1833, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_half_kernel_data(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6933, round(indices['auc'], 4))
        self.assertEqual(0.5228, round(indices['aupr'], 4))
        self.assertEqual(0.1833, round(indices['mcc'], 4))
