"""Dataset loaders that turn external corpora into GRACE Task definitions.

Submodules:
* ``tml_bench``  -- TML-bench (github.com/mykolapinchuk/tml-bench): recent Kaggle
  tabular competitions selected for contamination control (post-cutoff release).
* ``tabred``     -- TabReD (github.com/yandex-research/tabred): 8 industry-grade
  tabular datasets (NeurIPS D&B 2024).
* ``synthetic``  -- make_classification / make_regression with a documented DGP
  and ground-truth informative features (AssistedDS-style).
* ``reference``  -- compute baseline_score / oracle_score for FeatEng-style
  normalization.
"""
