import unittest

from sklearn.datasets import load_iris

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant


class TestCommunitySeparability(unittest.TestCase):
    def test(self):
        dataset = load_iris()

        evaluation = community_separability.compute_separability(
            embedding=dataset.data,
            communities=dataset.target,
            variant=SeparabilityVariant.TSPS)

        self.assertDictEqual(evaluation, {})
