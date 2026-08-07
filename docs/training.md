# The loop: data → train → eval → serve

This is the page to read with the cockpit open. Everything below happens in a project;
nothing exists until you create one.

## 1. Create a project

Projects tab → New. A project owns its classes (one entry for a plain counter; two
classes is most of what "multi-class" means — the split writes them into `data.yaml`),
its image-tag vocabulary, its dataset tree, its bots, and its owner (you). The labelling
style field ("dots" / "boxes") is recorded on the card; the editor itself places centres
and can size boxes in either case.

## 2. Get images in

Three routes, mix freely:

- **Upload** (Images tab). If the project already has a trained run, uploads can be
  pre-labelled by the *current* model as a draft — you fix, rather than start from
  nothing. Uploads land in the fix queue, never directly in training.
- **A Telegram bot** ([bots.md](bots.md)): every counted photo plus its ✓/✗ verdict
  flows into the pending inbox and fix queue. This is how a deployed counter keeps
  feeding its own training set.
- **The auto-labeler extra** (optional, license-gated download): a large
  vision-language model proposes labels for un-labelled images, which you then correct.
  Machine labels are drafts everywhere; a human pass is what promotes them.

## 3. Fix labels in the editor

Open any image from the Images tab. The editor is the one place labels change:

- **Mouse:** left-click adds a point, right-click removes, wheel or ± zooms,
  **← / →** moves to the previous/next image, **Esc** closes.
- **Touch:** pinch zooms, one-finger drag pans. A tap never changes the count — tap
  *selects* a point, long-press *adds* one, and the thumb pad acts on the selection
  (nudge arrows, delete). Fingers are wider than objects; select-then-act is what makes
  precision possible.
- **Boxes mode (⬚):** select a detection and press **R** (or use the handles) to size
  its box when extent matters, not just the centre.
- **Classes:** in a multi-class project, the class chips choose what a new point is.
- Counts and the save state are always visible; saving writes the label file. If you
  script against the editor, verify by re-reading the label file, never by looking at
  the canvas.

Tag images as you go (the bucket chips): tags are free-text properties of an *image*
("cluttered", "low-light"), never seen by the model, and the eval will later group
holdout error by them — so one hard image type cannot hide inside a blended average.

## 4. Decide each image's side: training or holdout

From the fix queue an image is **promoted** — moved, not copied — into either:

- **`raw/`, the training pool**, or
- **`testset/`, the frozen holdout.** Frozen means: never trained on, never synthesized
  from, and an image never comes back across. The holdout is the exam every run takes;
  its stability is what makes any two scores in the ledger comparable at all.

You can also **earmark** an image for the holdout in advance (the bot's 🎯 button does
this) — the earmark travels with the image so the split-side decision is already made
when it clears the fix queue. Aim for the holdout to look like *deployment*, including
the hard cases; a soft exam saturates (every run scores 0.0) and stops being able to
rank models — the eval will tell you when ties say that has happened.

## 5. Train

One control: **model × machine × card** (Train tab). `GET /api/train/options` answers
up front what may run where, so the UI disables a combination *with its reason* instead
of letting you press Train and read a 400. The answers are measured, not assumed:

- **Can this card run this family?** A family's venv carries a torch build with a fixed
  kernel set; a card whose architecture is missing from it cannot run that venv at all.
  Probed per machine, recorded, and re-asked when hardware changes.
- **Can this card *hold* this configuration?** Peak VRAM is measured from actual
  training runs at your requested size and batch — not the family default, and not a
  spec sheet. A configuration that has already OOM'd on a card this size is refused with
  that evidence. An undersized card is refused by default because (on WSL2 especially)
  overflow does not error — it silently spills into host RAM at ~3× the time cost.
  `allow_spill` overrides it deliberately.
- **Is the machine already training?** Refused by default; `allow_concurrent` (a
  checkbox) overrides it *by name*. Two GPU jobs at once has failed more often than it
  has worked on real hardware — a wedged driver call can hold a card in an unkillable
  state — so concurrency is an experiment you opt into, not a default you discover.
- **Remote runs sync first.** The project's dataset is rsynced to the worker before
  every remote run (seconds when nothing changed), and the launch is refused if the sync
  fails — training on data you could not confirm is the named worst failure mode.

Options you'll actually use: `split_seed` (train/val membership is a seeded hash of each
image name — same seed, same split, on any machine), a second size for back-to-back
training of the champion at e.g. 960 and 1280, and epochs/batch with per-family
defaults from the registry.

## 6. Read the results

The Results tab ranks runs by **count MAE (per image)** on the frozen holdout — the
product metric. mAP is recorded as a secondary signal; it structurally cannot see
calibration failures, and a model can top the mAP table while being the worst counter
offered. For every run, the eval also shows:

- the **miscount list** — which holdout images it got wrong, by how much, signed;
- the served operating point (**conf/iou**) and where it was tuned;
- **ties** — how many sweep cells matched the winner's score. Many ties on many recent
  runs means the holdout has saturated and can no longer rank this family; the fix is
  harder holdout data, not more runs;
- **grid-edge** flags — the winner sat on the sweep boundary, so its score is a floor;
- per-tag MAE, using your bucket chips.

Then **open the eval preview gallery and look at the pictures.** Every holdout image gets
a preview with the run's detections on it. The two most consequential model bugs in this
system's history were found by *looking* — a model that invented an object on a permanent
scuff mark in the background, and a model that scored zero on the one image whose scene
looked different — and neither was visible in any aggregate number. Both were data
lessons, not code lessons: photograph what confuses the model, label it, train on it.

## 7. Pin the champion

By default a project serves its newest run. **Pinning** (Models list → activate) fixes
serving to a specific run — revert to last month's model in one click, no retraining.
The pin is per-project and per-machine, and never touched by training. If a challenger
family earns its place, the **engine** switch changes which family serves — it refuses
if that engine has no exported weights, and switching never disturbs the yolo pin, so
switching back is also one click.

## 8. Export

The Export tab turns a run into a deployable file. Formats depend on the family —
the champion exports ONNX, TorchScript, CoreML, TFLite, OpenVINO, NCNN and TensorRT;
challenger families offer the subset their tooling supports — and each format carries
its caveats in the UI rather than in a doc. Artifacts land under the run and are listed
with their sizes; anything served over HTTP is behind the same auth gate as everything
else.
