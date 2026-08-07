"""What ranks a run — one definition per metric, keyed by a project's choice.

`Project.headline_metric` has existed since Phase 1 and was read by NOTHING:
`_best_run` ranked on `mae` unconditionally, the cards said "MAE", and a project
that wanted mAP got a counting metric anyway. It is the third manifest field
found stored-but-inert (after `labelling` and `classes`), and the most
consequential, because it decides which run a project calls its best.

WHY IT CANNOT STAY count_mae FOR EVERYTHING. Count-MAE is mean absolute error in
OBJECTS PER IMAGE — it collapses an image to one number, which is exactly right for
"how many objects are in this image" and meaningless the moment a project has
several classes. Three bolts and one nut counted as four objects is a perfect
score for a model that cannot tell them apart. A multi-class project has to be
ranked on something that knows about classes, and mAP50 is the one the ledger
already records for every run.

DIRECTION IS PART OF THE METRIC, not a caller's problem. MAE is better lower and
mAP is better higher; a ranking function that assumes one of those silently
inverts the other, which would present the WORST run as the best with nothing
about it looking wrong. Each entry states its own direction and the tie-break
walks the same way for both.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    key: str
    field: str          # the ledger row's key
    label: str          # how a card should name it
    lower_is_better: bool
    # WHERE the number was measured. Not decoration: count-MAE is scored on the
    # FROZEN holdout by count_eval, and map50 is read out of ultralytics'
    # results.csv — the best epoch's score on the VAL SPLIT, which the model
    # early-stopped against. They are not equally trustworthy and a card that
    # showed them the same way would imply they were.
    source: str = "holdout"
    # A run missing this field cannot be ranked on it — it is not a zero, and
    # treating it as one would float unscored runs to the top of a "best" list.
    unit: str = ""

    def sort_key(self, row):
        """(value, -test_images, -finished) with value flipped when higher wins.

        The secondary keys are not decoration: the holdout is saturated, several
        runs score identically, and "most holdout images, then most recent" is
        what stops a 0.0 on 63 images outranking a 0.0 on 65 — a real wrong
        answer this file's caller used to give.
        """
        # A ROW WITHOUT THE METRIC SORTS LAST, rather than raising. An
        # unscored run has no value to rank on, and the documented guard
        # against that did not exist — `-v` on None was a TypeError that took
        # the whole comparison down.
        v = row.get(self.field)
        if v is None:
            return (float("inf"), 0, 0)
        v = v if self.lower_is_better else -v
        # test_images: the ledger's own column. This read `test_trays`, a key
        # nothing has ever written, so the tie-break was always 0 — exactly the
        # wrong ranking the docstring above says it prevents.
        return (v, -(row.get("test_images") or 0), -(row.get("finished") or 0))


METRICS: dict[str, Metric] = {
    "count_mae": Metric(key="count_mae", field="mae", label="MAE",
                        lower_is_better=True, unit="objects/image",
                        source="holdout"),
    # HONEST LABEL, and a known gap. This is max(metrics/mAP50(B)) from the run's
    # results.csv: the best epoch's mAP on the val split. The val split is what
    # ultralytics early-stops and picks best.pt against, so the number is
    # optimistic by construction — a model is being scored on data that chose it.
    # count_eval evaluates COUNT on the frozen holdout and does not evaluate mAP
    # at all, so there is no held-out mAP to use yet.
    #
    # Offered anyway because a multi-class project cannot be ranked on count-MAE
    # (three bolts and one nut counted as four objects is a perfect score for a
    # model that cannot tell them apart), and a val-split mAP beats a metric that
    # is structurally blind. It is labelled `val split` everywhere it appears so
    # nobody mistakes it for the holdout, and closing the gap means running a
    # validator against testset/ — see PIVOT Phase 3.
    "map50": Metric(key="map50", field="map50", label="mAP50",
                    lower_is_better=False, unit="", source="val split"),
    # The same measurement on the FROZEN holdout, by ultralytics' own validator
    # (eval_core.holdout_map). This is the one a multi-class project should rank
    # on: it is held out, and it knows about classes in a way count-MAE cannot.
    # Only runs scored since 2026-07-31 carry it — older ones simply do not
    # appear in a ranking that uses it, which is correct, because they have no
    # such number and inventing one would be the lie this metric exists to end.
    "map50_holdout": Metric(key="map50_holdout", field="map50_holdout",
                            label="mAP50", lower_is_better=False, unit="",
                            source="holdout"),
}

DEFAULT = "count_mae"


def get(key: str | None) -> Metric:
    """The metric a project ranks on, falling back to counting.

    An unknown key falls back rather than raising: a manifest naming a metric a
    later version removed should still list its runs, just ranked the old way.
    """
    return METRICS.get(key or DEFAULT, METRICS[DEFAULT])
