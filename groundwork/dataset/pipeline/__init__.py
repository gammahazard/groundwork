"""What a RUN is: split -> train -> score -> record -> (adopt).

    split.py             raw/ -> images|labels/{train,val} + data.yaml. Image-level
                         and STABLE (membership is a hash of the stem), so adding
                         an image moves only that image.
    train.py             ultralytics, FIXED 250 epochs, no early stopping — the
                         thing that makes two runs comparable at all.
    count_eval.py        the exam: conf x iou sweep on the FROZEN holdout, scored
                         as per-image count MAE. Writes count_eval.json, which is
                         what the runtime then serves its conf/iou from.
    run_snapshot.py      what this run actually consumed — manifest + content
                         addressed image blobs, so any run stays diffable and any
                         image restorable byte-exact.
    training_history.py  the ledger, one row per completed run, plus run_facts().
    adopt_run.py         bring a lab-trained run home: rsync, renumber, record.

ORDER IS LOAD-BEARING. run_snapshot writes the manifest that training_history's
run_facts() reads for the ledger's image/object counts, so the snapshot runs BEFORE
the ledger write (retrain_job.run_pipeline). Reverse them and every row silently
records the live dataset instead of the run's own.

Four of these are launched as SUBPROCESSES by web/retrain_job.py, not imported.
Their module paths live in one place there (`_STAGE`), because cancel()'s
last-resort pkill has to match the same strings the launches use — two lists
that drift means a cancel leaves a stage running while the state says
"cancelled". `adopt_run` is additionally the ExecStart of the adopt job's systemd
unit on HQ. Moving anything in this package means editing those too; a re-export
shim does not help, because `python -m` resolves the real path.

This package deliberately does NOT re-export its modules: train imports
ultralytics and count_eval imports it lazily, so an eager re-export would drag
torch into anything that merely touched the package.
"""
from __future__ import annotations

__all__ = ["adopt_run", "count_eval", "run_snapshot", "split", "train",
           "training_history"]
