"use strict";
/* Cancel the run on ONE card. Split from train.js, which renders the cards.
 *
 * WHY A CONFIRM. Cancel is destructive and not undoable: it SIGKILLs the process
 * group and deletes the half-finished run directory, so an accidental click
 * costs however many epochs were in it. The dialog therefore names the run, the
 * machine, the card and how far along it was — enough to notice you are about to
 * kill the wrong one on a two-GPU box, which is the mistake per-card buttons
 * make possible in the first place.
 *
 * WHY IT SAYS "SCRAPPED". A cancelled run leaves nothing behind, and that is
 * deliberate: a half-trained directory with weights in it is indistinguishable
 * from a finished one in the Models list. The server only skips the delete when
 * the run got as far as writing count_eval.json, i.e. it is a real result. The
 * response reports which happened and this shows it back rather than assuming.
 *
 * ONE ENDPOINT FOR BOTH KINDS AND BOTH MACHINES: DELETE /api/train. The
 * machine/kind/card go in the query, and for a remote machine the server
 * forwards the identical request rather than mapping kinds to endpoints twice.
 */

window.trainCancel = async function (opts) {
  const {machineKey, machineName, card, kind, run, more, local} = opts;
  const what = kind === "yolo" ? "yolo retrain" : "challenger run";
  if (!confirm(
      `Cancel the ${what} on ${machineName} card ${card}?\n\n`
    + `${run}${more ? ` — ${more}` : ""}\n\n`
    + `This kills it immediately and DELETES the half-finished run directory. `
    + `It cannot be undone.`
    + (local ? "" : `\n\n(Sent to ${machineName} over the network.)`))) return;

  const why = $("#tWhy");
  const say = t => { if (why) { why.textContent = t; why.hidden = false; } };
  say(`cancelling ${run}…`);

  /* MARK THE CARD, do not just disable the element. _paintBusy rebuilds the
   * panel with innerHTML on every heartbeat, so anything set on the button here
   * is gone within seconds — and the run is still listed, because the backend
   * stops it at the next stage boundary rather than instantly. The button
   * therefore reappeared enabled over a run that was already being cancelled,
   * which reads as "nothing happened" and gets clicked again. Reported exactly
   * that way. The mark lives in train.js and the RENDER honours it. */
  if (window.trainMarkCancelling) trainMarkCancelling(machineKey, card, true);
  try {
    const q = `machine=${encodeURIComponent(machineKey)}`
            + `&kind=${encodeURIComponent(kind)}`
            + (card == null ? "" : `&card=${card}`);
    const r = await apiOrThrow(`/api/train?${q}`, {method: "DELETE"});
    if (r.ok === false) {
      // Not an error: the run may simply have finished between the poll that
      // drew the button and the click.
      say(`nothing to cancel — ${r.error || "it is no longer running"}`);
    } else {
      const kept = r.scrapped === false
        ? " (kept: it had already been scored, so it is a real result)"
        : " and scrapped";
      say(`✓ cancelled ${r.run || run}${kept}.`);
    }
  } catch (e) {
    say(`✗ could not cancel: ${e.message || e}`);
    // It failed, so give the control straight back rather than making them wait
    // out the settle timeout.
    if (window.trainMarkCancelling) trainMarkCancelling(machineKey, card, false);
  }
  /* NOT CLEARED HERE ON SUCCESS. The request returning means the cancel was
   * ACCEPTED, not that the run has stopped — the pipeline honours it at its next
   * stage boundary. _paintBusy clears the mark when the job actually leaves the
   * busy list, which is the only moment that means it worked. Clearing it here
   * would hand back a live-looking button over a run still winding down. */
  if (window.trainBusyTick) trainBusyTick();
  if (window.loadYolo) loadYolo();
  if (window.loadLab) loadLab();
};
