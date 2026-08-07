/* Edit a project's class list from the cockpit, not from project.json.
 *
 * WHY IT LOOKS CAUTIOUS. A class is an INDEX, not a name. Every label file on
 * disk is `<cls> <cx> <cy> <w> <h>`, so this list is a SCHEMA for dots already
 * drawn, and rewriting it rewrites what they mean. The server enforces that
 * (project_classes.py); this UI's job is to make the safe edits obvious and the
 * dangerous ones visibly impossible, rather than letting you compose a request
 * that comes back 409.
 *
 *   append          a new row at the bottom      — always safe
 *   rename          type over the name           — safe, the index does not move
 *   reorder         NOT OFFERED                  — would re-point every label
 *   remove          only with the count at zero  — and the server re-checks
 *
 * NO DRAG HANDLES, NO MOVE ARROWS, deliberately. The one edit that silently
 * corrupts a dataset is the one this screen does not let you express.
 *
 * THE LABEL COUNT NEXT TO EACH CLASS IS THE POINT. It is what turns "can I
 * delete this?" from a guess into a fact, and it counts the frozen testset too —
 * a class used only there can never be re-labelled away.
 */
(function () {
  let SLUG = null, rows = [], busy = false;

  const esc = s => String(s).replace(/[&<>"]/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function render(note) {
    const host = document.getElementById('pjClassBody');
    if (!host) return;
    host.innerHTML = rows.map((r, i) => `
      <div class="pjClsRow">
        <span class="pjClsIdx">${i}</span>
        <input class="pjClsName" data-i="${i}" value="${esc(r.name)}"
               ${busy ? 'disabled' : ''} spellcheck="false">
        <span class="pjClsUse">${r.labels ? r.labels.toLocaleString() + ' labels'
                                          : r.isNew ? 'new' : 'unused'}</span>
        ${r.labels || rows.length < 2 ? ''
          : `<button class="pjClsDel" data-i="${i}" title="Remove this class"
                     ${busy ? 'disabled' : ''}>remove</button>`}
      </div>`).join('') + `
      <div class="pjClsAdd">
        <input id="pjClsNew" placeholder="add a class…" spellcheck="false"
               ${busy ? 'disabled' : ''}>
        <button id="pjClsAddBtn" ${busy ? 'disabled' : ''}>add</button>
      </div>
      <p class="pjSub" id="pjClsNote">${note || ''}</p>`;

    host.querySelectorAll('.pjClsName').forEach(el => {
      el.oninput = () => { rows[+el.dataset.i].name = el.value; };
    });
    host.querySelectorAll('.pjClsDel').forEach(el => {
      el.onclick = () => {
        // Only ever the LAST rows can go without renumbering the others, and
        // the server refuses anything else — so removing from the middle would
        // be offering an edit that cannot succeed.
        const i = +el.dataset.i;
        if (i !== rows.length - 1) {
          return note1('only the last class can be removed — removing one from '
                     + 'the middle would renumber every label after it');
        }
        rows.splice(i, 1);
        render();
      };
    });
    const add = () => {
      const inp = document.getElementById('pjClsNew');
      const v = (inp.value || '').trim();
      if (!v) return;
      if (rows.some(r => r.name === v)) return note1(`"${v}" is already a class`);
      rows.push({ name: v, labels: 0, isNew: true });
      render();
      document.getElementById('pjClsNew').focus();
    };
    const btn = document.getElementById('pjClsAddBtn');
    if (btn) btn.onclick = add;
    const inp = document.getElementById('pjClsNew');
    if (inp) inp.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); add(); } };
  }

  function note1(msg) {
    const n = document.getElementById('pjClsNote');
    if (n) n.textContent = msg;
  }

  async function save() {
    const names = rows.map(r => (r.name || '').trim()).filter(Boolean);
    if (!names.length) return note1('a project needs at least one class');
    busy = true; render('saving…');
    // allow_removal is sent only when the list actually got shorter, so the
    // server's confirm-step still guards a typo that deletes a row.
    const shrank = names.length < (window.__pjClassCount || names.length);
    let r, body;
    try {
      r = await fetch(`/api/projects/${encodeURIComponent(SLUG)}/classes`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classes: names, allow_removal: shrank })
      });
      body = await r.json();
    } catch (e) {
      busy = false; return render('could not reach the server');
    }
    busy = false;
    if (!r.ok) return render(body.detail || `refused (${r.status})`);
    await load(SLUG, 'saved · the next split rebuilds data.yaml');
    // The project bar prints the class list; leave it agreeing with the truth.
    if (window.loadProjects) window.loadProjects();
  }

  async function load(slug, note) {
    SLUG = slug;
    const r = await fetch(`/api/projects/${encodeURIComponent(slug)}/classes`);
    const b = await r.json();
    rows = (b.classes || []).map(c => ({ name: c.name, labels: c.labels }));
    window.__pjClassCount = rows.length;
    let n = note || '';
    if ((b.orphan_indices || []).length) {
      n = `⚠ labels use class ${b.orphan_indices.join(', ')}, which this project `
        + `does not declare — the next split will refuse until that is fixed`;
    }
    render(n);
  }

  window.openClassEditor = async function (slug) {
    let m = document.getElementById('pjClassModal');
    if (!m) {
      m = document.createElement('div');
      m.id = 'pjClassModal';
      m.className = 'modal';
      m.innerHTML = `<div class="modalCard pjClsCard">
        <button class="modalClose" id="pjClsX">✕</button>
        <h3 class="modalTitle">Classes</h3>
        <div id="pjClassBody"></div>
        <p class="pjSub">Renaming is safe — the index never moves. Reordering is
          not offered: it would silently re-point every label already drawn.
          A class can only be removed once nothing is labelled with it, counting
          the frozen testset.</p>
        <div class="pjClsFoot">
          <button id="pjClsCancel">Close</button>
          <button id="pjClsSave" class="primary">Save classes</button>
        </div></div>`;
      document.body.appendChild(m);
      const close = () => { m.remove(); };
      m.querySelector('#pjClsX').onclick = close;
      m.querySelector('#pjClsCancel').onclick = close;
      m.onclick = e => { if (e.target === m) close(); };
      m.querySelector('#pjClsSave').onclick = save;
      document.addEventListener('keydown', function esc2(e) {
        if (e.key === 'Escape' && document.getElementById('pjClassModal')) {
          close(); document.removeEventListener('keydown', esc2);
        }
      });
    }
    await load(slug);
  };
})();
