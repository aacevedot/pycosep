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

        self.assertEqual(1.0, indices['auc'])
        self.assertEqual(1.0, indices['aupr'])
        self.assertEqual(1.0, indices['mcc'])
