import unittest

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant
from tests.test_data import _half_kernel, _parallel_lines, _circles, _rhombus, _spirals


class TestLDPSCommunitySeparability(unittest.TestCase):
    def test_ldps_returns_expected_indices_when_half_kernel_data(self):
        embedding, communities = _half_kernel()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.7067, round(indices['auc'], 4))
        self.assertEqual(0.5421, round(indices['aupr'], 4))
        self.assertEqual(0.1833, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_circles_data(self):
        embedding, communities = _circles()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.5100, round(indices['auc'], 4))
        self.assertEqual(0.6425, round(indices['aupr'], 4))
        self.assertEqual(0.0000, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_rhombus_data(self):
        embedding, communities = _rhombus()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(1.0000, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_spirals_data(self):
        embedding, communities = _spirals()

        indices, _ = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        self.assertEqual(0.6128, round(indices['auc'], 4))
        self.assertEqual(0.6192, round(indices['aupr'], 4))
        self.assertEqual(0.1780, round(indices['mcc'], 4))

    def test_ldps_returns_expected_indices_when_parallel_lines_data(self):
        embedding, communities = _parallel_lines()

        indices, metadata = community_separability.compute_separability(
            embedding=embedding,
            communities=communities,
            variant=SeparabilityVariant.LDPS)

        # MATLAB: self.assertEqual(1.0000, round(indices['auc'], 4))
        self.assertEqual(0.5833, round(indices['auc'], 4))
        # MATLAB: self.assertEqual(1.0000, round(indices['aupr'], 4))
        self.assertEqual(0.6199, round(indices['aupr'], 4))
        # MATLAB: self.assertEqual(1.0000, round(indices['mcc'], 4))
        self.assertEqual(0.0000, round(indices['mcc'], 4))
