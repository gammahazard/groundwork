"use strict";
/* Add images to a project, from the machine you are sitting at.
 *
 * Every image in the first project arrived through the Telegram bot, which is exactly
 * right for it and useless for a project that has no bot yet — which is every
 * project on the day it is created. Phase 6 is not finished by being able to
 * CREATE a project; it is finished when you can put data in one.
 *
 * ONE MODAL, THREE DOORS. The button sits on Fix queue, Training set and Test
 * set, because those are the three places you are standing when you think "I
 * should add some images". They all open the same dialog and all land in the
 * FIX QUEUE — and the dialog SAYS so, because a button on the Test set tab that
 * quietly puts images somewhere else is worse than no button. `raw/` and
 * `testset/` mean labelled; an unlabelled image in either would train on an
 * empty label file and score zero with nothing about it looking wrong.
 *
 * PRE-LABELLING IS OPTIONAL AND HONEST ABOUT ITSELF. Ticking the box runs the
 * project's SERVING model over each image first and saves its boxes as a draft
 * (dataset/store/prelabel.py) — minutes of fixing instead of an hour of
 * clicking. Still a draft: it lands in the fix queue like everything else,
 * because promoting model output unreviewed is how a counting error becomes a
 * permanent one. The checkbox NAMES the model it would use and disables itself
 * when the project has none, rather than offering a tick that does nothing.
 *
 * PER-FILE OUTCOMES ARE SHOWN, not collapsed into one message. Dropping twenty
 * photos in and being told "1 failed" is useless: which one, and why? The
 * server reports each skip with a reason (not an image, empty, too large,
 * unusable filename) and this renders them, because the answer to "why is my
 * image count 19" should not require reading a log.
 */

const _uEsc = t => String(t ?? "").replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

let UP_FILES = [];
const UP_HINT = "JPEG · PNG · WEBP · HEIC · up to 60MB each";

function _upOpen() {
  const m = $("#upModal");
  if (!m) return;
  UP_FILES = [];
  $("#upFiles").value = "";
  $("#upPicked").textContent = UP_HINT;
  $("#upGo").disabled = true;
  $("#upGo").textContent = "Upload";
  $("#upErr").hidden = true;
  $("#upReport").hidden = true;
  m.hidden = false;
  _upModelName();
}

function _upClose() {
  const m = $("#upModal");
  if (m) m.hidden = true;
}

/* WHICH model would draw the dots. Named, not implied: "the pinned model" is
 * useless when you have four runs and cannot remember which is serving, and a
 * project with none must not offer a checkbox that silently does nothing. */
async function _upModelName() {
  const el = $("#upPredictWho"), box = $("#upPredict");
  if (!el) return;
  el.textContent = "checking which model is serving…";
  try {
    const s = await apiOrThrow("/api/state");
    const name = s.served && s.served.name;
    if (name) {
      const mae = s.served.mae != null ? ` · holdout MAE ${fmtScore(s.served.mae)}` : "";
      el.textContent = `${name}${mae} — you still check its work in the fix queue`;
      if (box) box.disabled = false;
    } else {
      el.textContent = "this project has no model serving yet — train one first";
      if (box) { box.checked = false; box.disabled = true; }
    }
  } catch (e) {
    el.textContent = "could not tell which model is serving";
    if (box) { box.checked = false; box.disabled = true; }
  }
}

function _upPick(files) {
  UP_FILES = [...(files || [])];
  const n = UP_FILES.length;
  $("#upPicked").textContent = n
    ? `${n} file${n === 1 ? "" : "s"}: ` + UP_FILES.slice(0, 3).map(f => f.name).join(", ")
      + (n > 3 ? ` and ${n - 3} more` : "")
    : UP_HINT;
  $("#upGo").disabled = !n;
}

async function uploadFiles(files, predict) {
  const report = $("#upReport"), err = $("#upErr"), go = $("#upGo");
  if (!files || !files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  // FormData values are strings on the wire; FastAPI's Form(bool) reads
  // "true"/"false", so send the word rather than a checkbox's "on".
  fd.append("predict", predict ? "true" : "false");
  go.disabled = true;
  go.textContent = predict ? "uploading + labelling…" : "uploading…";
  err.hidden = true;
  report.hidden = true;
  try {
    // apiOrThrow carries ?project= and turns a 4xx into a real error — a plain
    // fetch would resolve on 422 and render "undefined".
    const r = await apiOrThrow("/api/upload", {method: "POST", body: fd});
    const n = r.added.length;
    const drew = r.added.reduce((a, x) => a + (x.predicted || 0), 0);
    let head = `<b>${n} image${n === 1 ? "" : "s"} added to the fix queue</b>`;
    if (r.predict && drew)
      head += ` — the model drew <b>${drew}</b> object${drew === 1 ? "" : "s"} to check`;
    if (r.predict_error)
      head += ` <span class="short">(${_uEsc(r.predict_error)})</span>`;
    report.innerHTML = head
      + (r.skipped.length
          ? `<ul class="upSkips">` + r.skipped.map(s =>
              `<li><b>${_uEsc(s.name)}</b> — ${_uEsc(s.why)}</li>`).join("") + `</ul>`
          : "")
      + `<div class="upActs"><button id="upToFix">Go and label them →</button></div>`;
    report.hidden = false;
    const jump = $("#upToFix");
    if (jump) jump.onclick = () => {
      _upClose();
      showTab("images");
      // …and select the pane, or the tab opens on whichever was last shown and
      // the uploads you just made are on a different one.
      if (window.imagesShow) imagesShow("fix", {reload: true});
    };
    // The badge and the grid both change; refresh whichever is on screen.
    if (window.loadState) loadState();
    if (document.querySelector("#fix.active")) loadGrid("needs_fix", "#fixGrid");
    UP_FILES = [];
    $("#upFiles").value = "";
    $("#upPicked").textContent = UP_HINT;
  } catch (e) {
    err.textContent = e.message || String(e);
    err.hidden = false;
  }
  go.disabled = !UP_FILES.length;
  go.textContent = "Upload";
}

(() => {
  if (!document.querySelector("#upModal")) return;
  document.querySelectorAll(".upOpen").forEach(b => { b.onclick = _upOpen; });
  $("#upClose").onclick = _upClose;
  $("#upCancel").onclick = _upClose;
  $("#upModal").addEventListener("click", e => {
    if (e.target.id === "upModal") _upClose();     // click the backdrop to close
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !$("#upModal").hidden) _upClose();
  });

  const drop = $("#upDrop"), input = $("#upFiles");
  input.onchange = () => _upPick(input.files);
  // Drag and drop, because the label says "drop them here" and that has to be
  // true. dragover MUST be prevented or the browser navigates to the file.
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.remove("over");
  }));
  drop.addEventListener("drop", e => {
    const fs = [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith("image/"));
    if (fs.length) _upPick(fs);
  });

  $("#upGo").onclick = () => uploadFiles(UP_FILES, $("#upPredict").checked);
})();
