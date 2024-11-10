# Travelling salesman path - Python wrapper

*This Python package is based on the MATLAB implementation available at [biomedical-cybernetics/travelling-salesman-path](https://github.com/biomedical-cybernetics/travelling-salesman-path).*

**Article Source:** [Geometric separability of mesoscale patterns in embedding representation and visualization of multidimensional data and complex networks](https://journals.plos.org/complexsystems/article?id=10.1371/journal.pcsy.0000012). Acevedo A, Wu Y, Traversa FL, Cannistraci CV (2024) Geometric separability of mesoscale patterns in embedding representation and visualization of multidimensional data and complex networks. PLOS Complex Systems 1(2): e0000012. [https://doi.org/10.1371/journal.pcsy.0000012](https://doi.org/10.1371/journal.pcsy.0000012)

## Before starting

This package requires [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) to execute calculations. Check the [requirements](https://github.com/aacevedot/pycosep/blob/main/REQUIREMENTS.md) and follow the instructions to install this dependency in your operating system.

## Installation

Run the following to install:

```shell
pip install pycosep
```

## Usage

Computing the community separability of a given embedding.

```python
from sklearn.datasets import load_iris

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant

# Sample data. Details at https://scikit-learn.org/stable/datasets/toy_dataset.html
data = load_iris()

# Community separability variant
#   CPS: centroid projection separability
#   LDPS: linear discriminant projection separability
#   TSPS: travelling salesman projection separability
variant = SeparabilityVariant.TSPS

indices, _ = community_separability.compute_separability(
    embedding=data.data,
    communities=data.target,
    variant=variant)

print(f"AUC index: {indices['auc']}")
print(f"AUPR index: {indices['aupr']}")
print(f"MCC index: {indices['mcc']}")
```

Computing the community separability of a given embedding for 1000 permutations.

```python
from sklearn.datasets import load_iris

from pycosep import community_separability
from pycosep.separability_variants import SeparabilityVariant

# Sample data. Details at https://scikit-learn.org/stable/datasets/toy_dataset.html
data = load_iris()

# Community separability variant
#   CPS: centroid projection separability
#   LDPS: linear discriminant projection separability
#   TSPS: travelling salesman projection separability
variant = SeparabilityVariant.LDPS

# Number of iterations for the Null model
permutations = 1000

index_permutations, _ = community_separability.compute_separability(
    embedding=data.data,
    communities=data.target,
    variant=variant,
    permutations=permutations)

auc_results = index_permutations['auc']
print(f"AUC original value: {auc_results['original_value']}")
print(f"AUC p-value: {auc_results['p_value']}")
print(f"AUC mean: {auc_results['mean']}")
print(f"AUC max: {auc_results['max']}")
print(f"AUC min: {auc_results['min']}")
print(f"AUC standard_deviation: {auc_results['standard_deviation']}")
print(f"AUC standard_error: {auc_results['standard_error']}")

aupr_results = index_permutations['aupr']
print(f"AUPR original value: {aupr_results['original_value']}")
print(f"AUPR p-value: {aupr_results['p_value']}")
print(f"AUPR mean: {aupr_results['mean']}")
print(f"AUPR max: {aupr_results['max']}")
print(f"AUPR min: {aupr_results['min']}")
print(f"AUPR standard deviation: {aupr_results['standard_deviation']}")
print(f"AUPR standard error: {aupr_results['standard_error']}")

mcc_results = index_permutations['mcc']
print(f"MCC original value: {mcc_results['original_value']}")
print(f"MCC p-value: {mcc_results['p_value']}")
print(f"MCC mean: {mcc_results['mean']}")
print(f"MCC max: {mcc_results['max']}")
print(f"MCC min: {mcc_results['min']}")
print(f"MCC standard deviation: {mcc_results['standard_deviation']}")
print(f"MCC standard error: {mcc_results['standard_error']}")
```

## Reporting an issue

For reporting problems, questions, and suggestions; please, use the [Issues](https://github.com/aacevedot/pycosep/issues) section.