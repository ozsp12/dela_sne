"""Reproduce manuscript experiments using the canonical installed LAC core.

The experiment body retains paper-specific diagnostics, datasets, plotting, and
initialization choices. Before execution, every LAC primitive used by that body
is rebound to the implementation in ``src/dela_sne``. This makes the
manuscript validate the published software rather than an independent copy of
the clustering updates. Scientific-revision diagnostics are then executed from
a separate paper-only module that also imports the canonical package.
"""

from __future__ import annotations

import _experiment_body as experiments
import revision_experiments
from dela_sne import assign_weighted, dispersions_from_labels, softmax_feature_weights
from dela_sne.lac_diagnostics import iterate_with_temperature_rule


def _bind_canonical_lac() -> None:
    experiments.dispersions_from_labels = dispersions_from_labels
    experiments.assign_weighted = assign_weighted
    experiments.softmax_feature_weights = softmax_feature_weights
    experiments._iterate_with_temperature_rule = iterate_with_temperature_rule


def main() -> None:
    _bind_canonical_lac()
    experiments.main()
    revision_experiments.main()


if __name__ == "__main__":
    main()
