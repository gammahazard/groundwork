"""Model families: what they are, where they run, and how to score them.

ONE registry (`registry.py`) replaces six hand-maintained tables that each knew
part of the answer — which venv trains a family, which predictor scores it,
which run-name prefixes belong to it, what it is licensed under. They drifted in
the way this repo's audit predicts: an unrecognised arch used to fall through to
RF-DETR's loader, so scoring a D-FINE run would have loaded a D-FINE checkpoint
with the wrong loader from a venv that cannot even train it.

The invariant this package exists to protect: a model family is described in
exactly one place, and anything that needs to dispatch on one asks here.
"""
