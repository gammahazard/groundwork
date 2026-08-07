"use strict";
/* Add, rename and reorder a project's IMAGE TAGS from the cockpit.
 *
 * This existed only as a hardcoded array in buckets.js, so adding "scuffed" or
 * "low-light" meant editing JavaScript and redeploying. The server never needed
 * that list — bucket_mae groups by whatever tags exist — so it was pure UI.
 *
 * A TAG IS NOT A CLASS, and the safety rules are the OPPOSITE way round. Worth
 * stating plainly because the two editors look similar and behave differently:
 *
 *   a CLASS is an INDEX baked into every label file, so reordering silently
 *   re-points every dot ever drawn. classes.js therefore offers no way to
 *   reorder at all.
 *
 *   a TAG is a free-text STRING in {stem: tag}. Reordering is harmless AND
 *   meaningful — it sets the order the chips read in, deliberately a
 *   simple -> hard difficulty ramp — so this editor DOES offer it.
 *
 * The real hazard here is orphaning, not corruption: rename "stacked" and every
 * image still carrying that string drops out of the vocabulary. So a rename
 * MIGRATES the stored tags server-side, and the count beside each tag says how
 * many images a removal would strand.
 */
(function () {
  let rows = [], origin = [], collection = null, busy = false;

  const esc = s => String(s).replace(/[&<>"]/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function render(msg) {
    const host = document.getElementById('tagEdBody');
    if (!host) return;
    host.innerHTML = rows.map((r, i) => `
      <div class="tagEdRow">
        <span class="tagEdMove">
          <button data-up="${i}" ${i === 0 ? 'disabled' : ''} title="Earlier — simpler">↑</button>
          <button data-dn="${i}" ${i === rows.length - 1 ? 'disabled' : ''} title="Later — harder">↓</button>
        </span>
        <input class="tagEdName" data-i="${i}" value="${esc(r.name)}"
               ${busy ? 'disabled' : ''} spellcheck="false">
        <span class="tagEdUse">${r.images ? r.images + ' image' + (r.images === 1 ? '' : 's')
                                         : (r.isNew ? 'new' : 'unused')}</span>
        <button class="tagEdDel" data-del="${i}" ${busy || rows.length < 2 ? 'disabled' : ''}
                title="${r.images ? 'These images keep this tag and still group under it'
                                 : 'Remove this tag'}">×</button>
      </div>`).join('') + `
      <div class="tagEdAdd">
        <input id="tagEdNew" placeholder="add a tag… (e.g. scuffed, low-light, cluttered)"
               spellcheck="false" ${busy ? 'disabled' : ''}>
        <button id="tagEdAddBtn" ${busy ? 'disabled' : ''}>add</button>
      </div>
      <p class="hint" id="tagEdNote">${msg || ''}</p>`;

    host.querySelectorAll('.tagEdName').forEach(el => {
      el.oninput = () => { rows[+el.dataset.i].name = el.value; };
    });
    host.querySelectorAll('[data-up]').forEach(b => b.onclick = () => {
      const i = +b.dataset.up;
      [rows[i - 1], rows[i]] = [rows[i], rows[i - 1]];
      render();
    });
    host.querySelectorAll('[data-dn]').forEach(b => b.onclick = () => {
      const i = +b.dataset.dn;
      [rows[i + 1], rows[i]] = [rows[i], rows[i + 1]];
      render();
    });
    host.querySelectorAll('[data-del]').forEach(b => b.onclick = () => {
      const i = +b.dataset.del, r = rows[i];
      // Removing a tag that images carry does not untag them — it drops the tag
      // from the dropdown while those images keep the string and keep grouping
      // under it in count_eval. Say so rather than letting it look destructive.
      if (r.images && !confirm(
        `${r.images} image(s) are tagged "${r.name}".\n\n`
        + `Removing it from the list does NOT untag them — they keep the tag and `
        + `still group under it in run reports. It just stops being offered.\n\n`
        + `Remove it from the list?`)) return;
      rows.splice(i, 1);
      render();
    });
    const add = () => {
      const inp = document.getElementById('tagEdNew');
      const v = (inp.value || '').trim();
      if (!v) return;
      if (v === 'untagged') return note('"untagged" is what an untagged image already reads as');
      if (rows.some(r => r.name === v)) return note(`"${v}" is already a tag`);
      rows.push({ name: v, images: 0, isNew: true });
      render();
      document.getElementById('tagEdNew').focus();
    };
    const btn = document.getElementById('tagEdAddBtn');
    if (btn) btn.onclick = add;
    const inp = document.getElementById('tagEdNew');
    if (inp) inp.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); add(); } };
  }

  function note(msg) {
    const n = document.getElementById('tagEdNote');
    if (n) n.textContent = msg;
  }

  async function save() {
    const names = rows.map(r => (r.name || '').trim()).filter(Boolean);
    if (!names.length) return note('a project needs at least one tag');
    busy = true; render('saving…');
    let r, body;
    try {
      r = await fetch(withProject('/api/bucket-vocab'), {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ buckets: names })
      });
      body = await r.json();
    } catch (e) { busy = false; return render('could not reach the server'); }
    busy = false;
    if (!r.ok || body.ok === false) return render(body.error || `refused (${r.status})`);

    // THE MODAL STAYS OPEN so the outcome can be read. A rename silently
    // re-tags images server-side and a removal can strand them — closing on
    // success would make both invisible, which is the opposite of what an edit
    // with side effects should do.
    let msg = 'saved';
    if (body.migrated) msg += ` · ${body.migrated} image(s) re-tagged by the rename`;
    const stray = Object.keys(body.stranded || {});
    if (stray.length) {
      msg += ` · ⚠ still on images but no longer offered: ${stray.join(', ')}`;
    }
    // The chips AND every image's dropdown are built from the vocabulary, so both
    // are rebuilt — not just the row that was clicked.
    await loadBucketVocab();
    if (window.loadGrid && collection && TAGGABLE[collection])
      await loadGrid(collection, GRID_OF[collection]);
    // Re-read usage so the counts beside each tag reflect any migration.
    try {
      const v = await api('/api/bucket-vocab');
      const usage = v.usage || {};
      rows = (v.buckets || []).map(b => ({ name: b, images: usage[b] || 0 }));
    } catch (e) { /* keep what is on screen */ }
    render(msg);
  }

  function close() {
    const m = document.getElementById('tagEdModal');
    if (m) m.remove();
  }

  window.openTagEditor = async function (coll) {
    collection = coll;
    let v = null;
    try { v = await api('/api/bucket-vocab'); } catch (e) { v = null; }
    if (!v) return;
    const usage = v.usage || {}, orphans = v.orphans || {};
    rows = (v.buckets || []).map(b => ({ name: b, images: usage[b] || 0 }));
    origin = rows.map(r => r.name);

    let m = document.getElementById('tagEdModal');
    if (!m) {
      m = document.createElement('div');
      m.id = 'tagEdModal';
      m.className = 'modal';
      m.innerHTML = `<div class="modalCard tagEdCard">
        <button class="modalClose" id="tagEdX">✕</button>
        <h3 class="modalTitle">Image tags</h3>
        <div id="tagEdBody"></div>
        <p class="hint">Tags are an <b>inventory and eval lens</b> — they never
          touch the model. On the holdout they split MAE per tag, so one object
          type regressing cannot hide inside a blended average. Order is the
          difficulty ramp the chips read in. Renaming re-tags the images that
          carry it.</p>
        <div class="tagEdFoot">
          <button id="tagEdCancel">Close</button>
          <button id="tagEdSave" class="primary">Save tags</button>
        </div></div>`;
      document.body.appendChild(m);
      m.querySelector('#tagEdX').onclick = close;
      m.querySelector('#tagEdCancel').onclick = close;
      m.onclick = e => { if (e.target === m) close(); };
      m.querySelector('#tagEdSave').onclick = save;
      document.addEventListener('keydown', function onEsc(e) {
        if (e.key === 'Escape' && document.getElementById('tagEdModal')) {
          close(); document.removeEventListener('keydown', onEsc);
        }
      });
    }
    const stray = Object.keys(orphans);
    render(stray.length
      ? `⚠ images carry tags not in this list: ${stray.join(', ')} — add them here `
        + `to make them selectable again`
      : '');
  };
})();
