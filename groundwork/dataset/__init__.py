"""The dataset side: how images become labels, and labels become a model.

The old docstring here described a single line —
`autolabel -> labelImg -> split -> train -> count_eval` — which stopped being
true a long time ago. labelImg is not used at all, and the graph now has THREE
roots feeding one training path:

    bot / cockpit feedback ─┐
    teacher/autolabel ──────┼─> raw/ ─> split ─> train ─> count_eval ─> serve
    synth/compose ──────────┘          (holdout scored every run)

Packages, and the one idea each exists to hold:

    filters/   the eval<->runtime filter set, and the invariant that they must
               MATCH — the project's most-repeated rule, given one home
    store/     reading and writing image data; collections are exclusive buckets
               and an image moves by FILE MOVE, which is what makes "never on both
               sides of train/test" true by construction
    teacher/   modules that label data for other models to train on
    synth/     synthetic images — wired, not currently in use

At this level, deliberately not in a package:

    paths      the directory layout. 29 import sites across 24 files in every
               subpackage — foundation, not storage.
    sidecar    atomic JSON for the files four processes share. Same reason.
    split, train, count_eval, run_snapshot, training_history, adopt_run
               the run pipeline. Grouping these is the next slice (REFACTOR.md);
               they carry the adopt job's systemd unit, so they move deliberately.
    point_to_box, yolo_export       point <-> box <-> YOLO-line conversions
    gallery, heatmap, timelapse, live_trainer, point_editor, dedup_check,
    pending_janitor                 review + visualisation tools

A module launched as `python -m groundwork.dataset.<name>` cannot be moved by a
re-export shim — the path is resolved for real. Check the invocation sites AND
the systemd units before moving one, and never while a retrain is running.
"""
