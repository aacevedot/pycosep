import unittest

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant
from tests.test_data import _half_kernel, _parallel_lines, _circles, _rhombus, _spirals


class TestCPSCommunitySeparability(unittest.TestCase):
    def test_cps_returns_expected_indices_when_half_kernel_data(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6933, round(indices['auc'], 4))
        self.assertEqual(0.5228, round(indices['aupr'], 4))
        self.assertEqual(0.1833, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_circles_data(self):
        embedding, communities = _circles()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.5100, round(indices['auc'], 4))
        self.assertEqual(0.6425, round(indices['aupr'], 4))
        self.assertEqual(0.0000, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_rhombus_data(self):
        embedding, communities = _rhombus()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_spirals_data(self):
        embedding, communities = _spirals()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6128, round(indices['auc'], 4))
        self.assertEqual(0.6132, round(indices['aupr'], 4))
        self.assertEqual(0.2527, round(indices['mcc'], 4))

    def test_cps_returns_expected_indices_when_parallel_lines_data(self):
        embedding, communities = _parallel_lines()

        indices, metadata = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.CPS)

        self.assertEqual(0.6528, round(indices['auc'], 4))
        self.assertEqual(0.6944, round(indices['aupr'], 4))
        self.assertEqual(0.3333, round(indices['mcc'], 4))
