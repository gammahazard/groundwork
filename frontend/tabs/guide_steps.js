"use strict";
/* What the Guide says. Content only — the stepper that shows it is guide.js.
 *
 * IT LIVES HERE AND NOT IN index.html because it was 166 lines of prose in the
 * middle of the page shell, and because prose needs editing far more often than
 * markup does. Keeping it as data also means the stepper can count the steps,
 * deep-link them and render them one at a time without parsing the DOM.
 *
 * AUDITED 2026-08-05 against what the cockpit actually is. Tonight's
 * restructure made four of its directions wrong, and a guide that sends you to
 * a tab that no longer exists is worse than no guide — it teaches the reader to
 * distrust it:
 *
 *   "Fix queue"        is a PANE inside Images now, not a tab
 *   "Models & Runs"    split into Train and Results
 *   "Count tab"        is hidden; it is not a place to send anyone
 *   "pin it" (step 7)  the Serving controls moved to the project Overview
 *
 * SCOPE, CORRECTED TOO. The old opening sold this as phone-and-chat — "on a
 * phone, in a chat, or from a camera". That undersells what it produces: the
 * output is a trained detector, a file, and it runs wherever you put it. The
 * object counter happens to run on a Raspberry Pi with a webcam; the phone and
 * the chat bot are two consumers of the same model, not the point of it.
 */

const GUIDE_STEPS = [
  {
    key: "project",
    title: "Make a project",
    lede: "A project is one thing you are teaching it to see, with its own data and its own labels.",
    body: `
      <p>Open <span class="gWhere">Projects</span> and press <b>+ New project</b>.
        It asks four things and nothing else:</p>
      <ul>
        <li><b>Name</b> — anything. The folder name is worked out from it and shown
          as you type, so it is never a surprise.</li>
        <li><b>What it detects</b> — your labels, comma separated. One
          (<code>object</code>) is the simple case; several (<code>bolt, nut</code>)
          makes it a multi-class project.</li>
        <li><b>Labelling</b> — <b>dots</b> means you click the centre of each object
          and its size is worked out from how close its neighbours are.
          <b>Boxes</b> means you draw the extent yourself.</li>
        <li><b>Ranked by</b> — which number decides your best attempt. Leave it on
          count&nbsp;MAE if you are counting.</li>
      </ul>`,
    note: {kind: "why", head: "Why this matters",
      text: `Count&nbsp;MAE is meaningless with several classes — three bolts and one
             nut counted as “four objects” is a perfect score for a model that
             cannot tell them apart. The form notices and suggests the other
             metric for you.`},
  },
  {
    key: "intake",
    title: "Get photographs in",
    lede: "Two ways in, and you can use both.",
    body: `
      <div class="gPair">
        <div><h4>Upload them</h4>
          <p>Drag files straight into the cockpit, on <span class="gWhere">Images</span>.
            Good for a folder you already have.</p></div>
        <div><h4>Send them from a phone</h4>
          <p>Set up a Telegram bot once in <span class="gWhere">Data collection</span> —
            paste a token from BotFather and choose which project it feeds. After that,
            photograph something and it arrives here.</p></div>
      </div>
      <p>Either way the photographs land in the <b>Fix queue</b> — the first of the
        three views on <span class="gWhere">Images</span>. It means exactly
        “arrived, not marked up yet”.</p>`,
    note: {kind: "care", head: "Careful",
      text: `A photograph is not training data until it is marked. An unmarked image
             is an empty answer sheet — train on it and the model learns that this
             picture contains nothing.`},
  },
  {
    key: "marking",
    title: "Mark what you want counted",
    media: {src: "/static/guide/marking.gif", alt: "The same image before and after marking — one dot per object",
            cap: "The same photograph, unmarked and marked. One click per object."},
    lede: "Open any photograph and click once on each object. That is the whole job.",
    body: `
      <ul>
        <li><b>Click</b> empty space to add · <b>right-click</b> a dot to remove.
          On a phone: <b>tap</b> to select, <b>press and hold</b> to add.</li>
        <li><span class="gKey">⬚&nbsp;Boxes</span> switches to sizing — drag a handle
          to set how big something is. You cannot add or remove dots while sizing,
          on purpose.</li>
        <li><span class="gKey">✛</span> asks <b>LocateAnything</b> for a second
          opinion: a large model reads the picture and guesses for you. Say what you
          are after (“object”, “bolt”, or your own words), wait 30–90 seconds, and its
          guesses appear as crosshairs. <b>Adopt</b> takes them as your dots.</li>
      </ul>`,
    note: {kind: "why", head: "Why this matters",
      text: `LocateAnything is how you avoid marking a hundred photographs by hand —
             let it guess, then fix what it missed. It never saves anything itself,
             so a bad guess costs you nothing. You still press <b>Save</b>.`},
  },
  {
    key: "split",
    title: "Send each photograph to one of two places",
    lede: "Once marked, a photograph goes to the training set or the holdout. Both live on Images.",
    body: `
      <ul>
        <li><b>Training set</b> — the model learns from these.</li>
        <li><b>Holdout</b> — the model <b>never</b> sees these while learning.
          They are the exam.</li>
      </ul>
      <p>Aim for roughly one in five in the holdout, and make them varied — the
        awkward ones, not just the tidy ones.</p>`,
    note: {kind: "why", head: "The one rule to keep",
      text: `A model can memorise anything it is shown. The only way to find out
             whether it has actually <i>learned</i> is to test it on photographs it
             has never seen. So the holdout is frozen: never trained on, and the
             same photograph never sits on both sides. Break that and every score
             afterwards is flattering and false.`},
  },
  {
    key: "training",
    title: "Train it",
    media: {src: "/static/guide/training.gif", alt: "Augmented training collages, several steps apart",
            cap: "What it is actually learning from — real collages, minutes apart in one run."},
    lede: "Pick a model, a machine and a card. Combinations that cannot work are greyed out with the reason.",
    body: `
      <p>Open <span class="gWhere">Train</span> and press <b>▶ Train</b>. It then runs
        on its own: it divides your training photographs into a learning set and a
        practice set, trains, and grades itself. Expect tens of minutes to a couple of
        hours. You can close the page.</p>
      <p>Press <span class="gKey">🔬 Watch training</span> to look inside while it
        works — the actual collages it is learning from, and the curves.</p>`,
    note: {kind: "why", head: "Why three choices and not one",
      text: `Not every model fits every card, and not every card can run every model.
             The picker knows both — which architectures a machine's environment was
             built for, and how much memory each model peaked at last time — so it
             can tell you why a combination is refused instead of failing an hour in.`},
  },
  {
    key: "verdict",
    title: "Read the verdict",
    media: {src: "/static/guide/verdict.jpg", alt: "Training curves for one run",
            cap: "One run's own curves. The score that ranks it is the count error, not these."},
    lede: "Every attempt is graded on the frozen holdout and lands in the table by itself.",
    body: `
      <p>Open <span class="gWhere">Results</span>. Which number ranks your runs is
        the <b>Ranked by</b> choice you made when you created the project, and the
        two answer different questions:</p>
      <ul>
        <li><b>count MAE</b> — on average, how many objects it gets wrong per
          photograph. <b>0.0 means it counted every holdout photograph exactly
          right</b>; 0.5 means it is off by half an object per photograph. This is
          the one to use when the job is <i>how many</i>, and the things you are
          counting are all the same kind.</li>
        <li><b>mAP50</b> — detection quality: did it find the right things, in the
          right places, with the right label. Higher is better. Use this when the
          <i>kind</i> matters — several classes, or objects of different shapes you
          need told apart.</li>
      </ul>
      <p>Pick by what a wrong answer would cost you. Miscounting is a count
        problem; confusing a bolt for a nut is not, and no counting score can
        see it.</p>`,
    note: {kind: "care", head: "Each one is blind to what the other measures",
      text: `<b>Count MAE cannot see class mistakes.</b> Three bolts and one nut
             counted as “four objects” is a perfect score for a model that cannot
             tell them apart — which is why a multi-class project should not be
             ranked on it.<br><br>
             <b>And mAP cannot see counting mistakes.</b> Measured here: a
             challenger scored the <i>best</i> mAP50 of any model tried
             (0.9780) and almost the worst count error — wrong on 27 of 68
             photographs, by up to 15 objects. mAP averages over many confidence
             thresholds; counting happens at exactly one, so a model can place
             boxes beautifully and still count badly. Rank a counter by mAP and
             you can ship the worst counter you have.`},
  },
  {
    key: "serving",
    title: "Put it to work",
    lede: "What you end up with is a file — a trained detector. It runs wherever you put it.",
    body: `
      <p>On the project's <span class="gWhere">Overview</span>, open
        <b>Change what serves</b> and pick the run you want. From then on it is the
        model that answers — in the chat bot, in this cockpit, and in anything you
        export. Changing your mind takes effect immediately.</p>
      <p>Nothing ties it to a phone. The object counter here also runs on a Raspberry&nbsp;5
        with a webcam over the image, counting live at a few frames a second. The same
        file would run on a laptop, a bench PC, or a camera on a line.</p>`,
    note: {kind: "care", head: "Where it runs is not where it learned",
      text: `A model trained on photographs taken by hand will be worse seen through a
             fixed overhead camera — it is a different viewpoint, and detectors do not
             transfer to one for free. If you deploy to a fixed camera, collect a few
             hundred photographs <i>from that camera</i> and train on those.`},
  },
  {
    key: "export",
    title: "Take it somewhere else",
    media: {src: "/static/guide/export.jpg",
            alt: "The Export tab: a run, a format, and the exports already built",
            cap: "Pick the run and the format. What you have already built is listed underneath."},
    lede: "Serving runs the model here. Exporting gives you a file to run anywhere else.",
    body: `
      <p>Open <span class="gWhere">Export</span>, pick a run and a format, and press
        <b>Export</b>. The conversion happens in the background — you can leave the
        page and come back to it.</p>
      <ul>
        <li><b>ONNX</b> — the neutral one. Almost every runtime reads it, and it is
          the usual first step into another format. Start here if you are unsure.</li>
        <li><b>CoreML</b> — iPhone, iPad and Mac, on the Neural&nbsp;Engine.</li>
        <li><b>TensorFlow&nbsp;Lite</b> — Android and small hardware.</li>
        <li><b>NCNN</b> — phones and a Raspberry&nbsp;Pi, no GPU needed.</li>
        <li><b>OpenVINO</b> — a machine with no GPU at all, on Intel chips.</li>
        <li><b>TorchScript</b> — PyTorch without Python, for a C++ or mobile runtime.</li>
        <li><b>TensorRT</b> — fastest on one specific NVIDIA card.</li>
      </ul>
      <p>The size is the run's own and is not a choice: exporting at another
        resolution is a different model, and one your holdout score no longer
        describes.</p>`,
    note: {kind: "care", head: "Two of them cost you something",
      text: `<b>NCNN</b> was measured here at 4.3&times; faster on a Pi and <i>one object
             worse</i> — fine for a live preview, not for a count that matters.
             <b>TensorRT</b> compiles for the exact card it runs on, so an engine built
             on one model of GPU will not load on another; export it on the card you
             will actually run it on. Both say so at the moment you pick them.`},
  },
  {
    key: "loop",
    title: "Make it better",
    lede: "The loop does not end. When the model gets something wrong, that photograph is the most valuable one you own.",
    body: `
      <ul>
        <li>Correct it in the <b>Fix queue</b> on <span class="gWhere">Images</span>
          and send it back to the training set.</li>
        <li>Add more photographs of whatever it struggles with.</li>
        <li>Train again and compare the two runs side by side.</li>
      </ul>`,
    note: {kind: "care", head: "When a perfect score stops meaning anything",
      text: `If every attempt scores 0.0, the exam has become too easy and can no
             longer tell your models apart. That is not success, it is a blunt
             instrument. The fix is harder holdout photographs — poor light, crowded,
             overlapping, awkward angles — added alongside the old ones so previous
             scores stay comparable.`},
  },
];

window.GUIDE_STEPS = GUIDE_STEPS;
