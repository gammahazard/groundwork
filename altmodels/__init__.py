"""Sidecar experiment: Apache-2.0 detectors scored by the SAME count-MAE exam
as the shipped ultralytics counter — without touching the existing flow.

Everything here is additive: artifacts under outputs/alt/ (invisible to the
runs ledger and to serving), heavy deps in separate venvs (.venv-alt /
.venv-mmdet), GPU shared via the repo's own gpu.lock. See docs/models.md
for the runbook and the plan for the gates (parity -> Gate A -> phone Gate B).
"""
