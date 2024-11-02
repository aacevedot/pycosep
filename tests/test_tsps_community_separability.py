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

    def test_tsps_returns_expected_indices_when_circles_data_without_permutations(self):
        embedding, communities = _circles()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_rhombus_data_without_permutations(self):
        embedding, communities = _rhombus()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(0.9100, round(indices['auc'], 4))
        self.assertEqual(0.9212, round(indices['aupr'], 4))
        self.assertEqual(0.4000, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_spirals_data_without_permutations(self):
        embedding, communities = _spirals()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(0.8614, round(indices['auc'], 4))
        self.assertEqual(0.8446, round(indices['aupr'], 4))
        self.assertEqual(0.5516, round(indices['mcc'], 4))

    def test_tsps_returns_expected_indices_when_parallel_lines_data_without_permutations(self):
        embedding, communities = _parallel_lines()

        indices, metadata = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.TSPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))
