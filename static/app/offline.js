// reading without a network (on a plane): a pack of the user's papers kept on the device, and
// the page that shows it. "Get ready" asks the server for the pack (/ui/offline: To read, the
// latest digest, the best of For you), waits for their AI summaries, and downloads the PDFs that
// enrich_daemon.py fetches for it (/offline/pdfs; arXiv gets one request per paper at its pace,
// so that takes minutes). everything goes into the Cache Storage PACK, which sw.js answers from
// when the server can't be reached. votes, read marks and notes made offline wait in lib.js's
// queue, and the pack on the device follows them (onPaperChange)
import { html, useState, useEffect, store, api, me, offline, unsynced, flushChanges, refreshMe, onPaperChange, canEdit, pageReady } from 'app/lib';
import { Paper, Page } from 'app/papers';

const PACK = 'pack-v1'; // as in sw.js
const PACK_URL = '/ui/offline';
const OPTS_KEY = 'resanity-pack-pdfs';
const PDF_POLL = 10000; // ms between asking how far the PDFs are
const PDF_TIMEOUT = 5 * 60000; // one PDF's download to the device
const DEVICE_TRIES = 3; // downloads of one PDF to the device before giving it up
const SUMMARY_ROUNDS = 60; // ~3 minutes of asking for AI summaries

const hasCaches = () => typeof caches !== 'undefined';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const pdfPath = pid => '/offline/pdf/' + pid;

// -----------------------------------------------------------------------------
// the pack on the device

export async function readPack() {
    if (!hasCaches()) return null;
    try {
        const r = await (await caches.open(PACK)).match(PACK_URL);
        return r ? await r.json() : null;
    } catch (e) {
        return null;
    }
}

let writing = Promise.resolve();
// one write at a time, each on top of the last: fn gets the stored pack and changes it
function updatePack(fn) {
    writing = writing.then(async () => {
        const pack = await readPack();
        if (!pack) return;
        fn(pack);
        await savePack(pack);
    }).catch(() => {});
    return writing;
}

async function savePack(pack) {
    const c = await caches.open(PACK);
    await c.put(PACK_URL, new Response(JSON.stringify(pack), { headers: { 'Content-Type': 'application/json' } }));
    packStore.set(pack);
}

async function pdfsOnDevice() {
    if (!hasCaches()) return new Set();
    const keys = await (await caches.open(PACK)).keys();
    return new Set(keys.map(k => new URL(k.url).pathname).filter(p => p.startsWith('/offline/pdf/')).map(p => p.slice(13)));
}

const allPapers = pack => pack.sections.flatMap(s => s.papers);

// a vote, read mark or note (sent or kept for later): the stored copy of the card follows
const without = (l, t) => (l || []).filter(x => x !== t);
const withTag = (l, t) => [...without(l, t), t];
const TAG_OPS = {
    add: (p, t) => { p.utags = withTag(p.utags, t); p.ntags = without(p.ntags, t); p.stags = without(p.stags, t); },
    sub: (p, t) => { p.utags = without(p.utags, t); },
    addneg: (p, t) => { p.ntags = withTag(p.ntags, t); p.utags = without(p.utags, t); p.stags = without(p.stags, t); },
    subneg: (p, t) => { p.ntags = without(p.ntags, t); },
    seen: (p, t) => { p.stags = withTag(p.stags, t); p.utags = without(p.utags, t); p.ntags = without(p.ntags, t); },
    unseen: (p, t) => { p.stags = without(p.stags, t); },
};
onPaperChange((path, form) => {
    const m = path.match(/^\/([a-z]+)\/(.+)$/);
    if (!m) return;
    const [, op, rest] = m;
    let pid = rest, apply;
    if (op === 'read' || op === 'unread') apply = p => { p.read = op === 'read'; };
    else if (op === 'note') apply = p => { p.note = String(form.text || '').trim(); };
    else if (TAG_OPS[op]) {
        const cut = rest.lastIndexOf('/'); // old arXiv ids have a slash: cs/0701001/<tag>
        pid = rest.slice(0, cut);
        const tag = decodeURIComponent(rest.slice(cut + 1));
        apply = p => TAG_OPS[op](p, tag);
    } else return;
    updatePack(pack => allPapers(pack).filter(p => p.id === pid).forEach(apply));
});

// -----------------------------------------------------------------------------
// getting ready: the pack, its summaries, its PDFs. runs on when the page is left (in the app)

const packStore = store(undefined); // undefined: not read yet
const job = store(null); // {step, summaries: [done, total], pdfs: {...}, error} while getting ready

const PDF_SECTIONS = [['toread', 'To read'], ['digest', 'the digest'], ['foryou', 'For you']];
const readOpts = () => {
    try { return JSON.parse(localStorage.getItem(OPTS_KEY)) || { toread: true, digest: true, foryou: false }; } catch (e) { return { toread: true, digest: true, foryou: false }; }
};
const opts = store(readOpts());
const setOpt = (k, v) => {
    const o = { ...opts.get(), [k]: v };
    opts.set(o);
    try { localStorage.setItem(OPTS_KEY, JSON.stringify(o)); } catch (e) { /* not remembered */ }
};

const patchJob = d => job.set({ ...(job.get() || {}), ...d });

async function getReady() {
    if (job.get() && job.get().running) return;
    job.set({ running: true, step: 'Putting your papers together…' });
    try {
        if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
        const pack = await api(PACK_URL);
        const old = await readPack();
        await savePack(pack);
        // the site itself, and cards that look the same offline: thumbnails and the math typesetting
        patchJob({ step: 'Saving the site and thumbnails…' });
        await keepApp(pack.warm);
        const c = await caches.open(PACK);
        const thumbs = allPapers(pack).map(p => p.thumb_url).filter(Boolean);
        await Promise.all(thumbs.map(u => fetch(u).then(r => r.ok && c.put(u, r)).catch(() => {})));
        // PDFs of the papers no longer in the pack leave the device
        if (old) {
            const keep = new Set(allPapers(pack).map(p => p.id));
            for (const pid of await pdfsOnDevice()) if (!keep.has(pid)) await c.delete(pdfPath(pid));
        }
        await Promise.all([summaries(pack), pdfs(pack)]);
        patchJob({ running: false, step: '' });
    } catch (e) {
        patchJob({ running: false, step: '', error: e.message });
    }
}

// the app's html and every file it loads, into sw.js's caches: written here, so they are there even
// if the service worker took over only after this page had loaded them
async function keepApp(extra) {
    const shell = await fetch('/', { cache: 'no-cache' });
    if (shell.ok) await (await caches.open('shell-v1')).put('/', shell);
    const urls = new Set(extra);
    document.querySelectorAll('link[rel="modulepreload"], script[type="module"][src], link[rel="stylesheet"]')
        .forEach(e => urls.add(e.getAttribute('href') || e.getAttribute('src')));
    try {
        Object.values(JSON.parse(document.querySelector('script[type="importmap"]').textContent).imports).forEach(u => urls.add(u));
    } catch (e) { /* the modulepreload links have them too */ }
    const c = await caches.open('static-v1');
    await Promise.all([...urls].map(async u => {
        if (await c.match(u, { ignoreVary: true })) return;
        const r = await fetch(u).catch(() => null);
        if (r && r.ok) await c.put(u, r);
    }));
}

async function summaries(pack) {
    const m = me.get();
    if (!m.summaries_ask) return;
    let missing = allPapers(pack).filter(p => !p.ai).map(p => p.id);
    const total = allPapers(pack).length;
    for (let round = 0; missing.length && round < SUMMARY_ROUNDS; round++) {
        patchJob({ summaries: [total - missing.length, total] });
        const got = {}, pending = [];
        for (let i = 0; i < missing.length; i += 60) {
            const res = await api('/summaries', { pids: missing.slice(i, i + 60).join(',') }, { keep: true });
            Object.assign(got, res.summaries);
            pending.push(...res.pending);
        }
        if (Object.keys(got).length) await updatePack(pk => allPapers(pk).forEach(p => { if (got[p.id]) p.ai = got[p.id]; }));
        missing = pending.filter(p => !got[p]);
        if (missing.length) await sleep(3000);
    }
    patchJob({ summaries: [total - missing.length, total] });
}

// the PDFs of the chosen sections: asked for, then downloaded one by one as the server has them
async function pdfs(pack) {
    if (!pack.pdfs_on) return;
    const o = opts.get();
    const want = [];
    for (const [key] of PDF_SECTIONS) {
        if (!o[key]) continue;
        for (const s of pack.sections) if (s.key === key) want.push(...s.papers.map(p => p.id));
    }
    const pids = [...new Set(want)].slice(0, pack.pdf_max);
    if (!pids.length) return;
    const res = await api('/offline/pdfs', { pids: pids.join(',') }, { keep: true });
    if (res.left_out) patchJob({ note: `${res.left_out} PDFs left out: at most ${pack.pdf_max} at a time.` });
    await downloadPdfs(pids);
}

// one PDF from the server into the pack, its progress told as it comes in; throws with what went wrong
async function savePdf(c, pid, size, say) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), PDF_TIMEOUT);
    try {
        const r = await fetch(pdfPath(pid), { signal: ctl.signal });
        if (!r.ok) throw new Error(`the server answered ${r.status}`);
        const total = Number(r.headers.get('Content-Length')) || size || 0;
        const chunks = [];
        let got = 0, told = 0;
        const reader = r.body.getReader();
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            got += value.length;
            if (Date.now() - told > 400) { told = Date.now(); say(got, total); }
        }
        await c.put(pdfPath(pid), new Response(new Blob(chunks, { type: 'application/pdf' }), {
            headers: { 'Content-Type': 'application/pdf', 'Content-Length': String(got) },
        }));
    } finally {
        clearTimeout(timer);
    }
}

const pdfProblem = e =>
    e.name === 'QuotaExceededError' ? 'This device has no room left for this site: free some space, or choose fewer PDFs.'
    : e.name === 'AbortError' ? `A PDF took over ${PDF_TIMEOUT / 60000} minutes to download; the connection may be too slow. It is tried again.`
    : `A PDF couldn't be saved on this device (${e.message}). It is tried again.`;

let pdfRun = 0; // a newer download loop ends the older one

// asks the server how far these papers' PDFs are, every PDF_POLL ms, and copies the ready ones to
// the device, one at a time, until none is left (a PDF that fails DEVICE_TRIES times is given up)
export async function downloadPdfs(pids) {
    const run = ++pdfRun;
    const c = await caches.open(PACK);
    const tries = {};
    for (;;) {
        const res = await api('/ui/offline/pdfs?pids=' + encodeURIComponent(pids.join(',')));
        if (run !== pdfRun) return;
        const have = await pdfsOnDevice();
        const count = st => pids.filter(p => res.pdfs[p] === st).length;
        const due = p => res.pdfs[p] === 'ready' && !have.has(p) && (tries[p] || 0) < DEVICE_TRIES;
        const state = () => ({
            device: pids.filter(p => have.has(p)).length, total: pids.length, waiting: count('waiting'),
            failed: count('failed'), stuck: pids.filter(p => res.pdfs[p] === 'ready' && !have.has(p) && tries[p] >= DEVICE_TRIES).length,
            wait: res.wait,
        });
        const todo = pids.filter(due);
        for (const [i, pid] of todo.entries()) {
            if (run !== pdfRun) return;
            const size = (res.sizes || {})[pid];
            const head = `Downloading PDF ${i + 1} of ${todo.length} to this device`;
            patchJob({ pdfs: state(), step: `${head}${size ? ` (${mb(size)})` : ''}…` });
            tries[pid] = (tries[pid] || 0) + 1;
            try {
                await savePdf(c, pid, size, (got, total) => run === pdfRun && patchJob({
                    step: `${head}: ${total ? `${Math.min(100, Math.round(100 * got / total))}% of ${mb(total)}` : mb(got)}…`,
                }));
                have.add(pid);
                patchJob({ pdfError: '' });
            } catch (e) {
                patchJob({ pdfError: pdfProblem(e) });
                if (e.name === 'QuotaExceededError') for (const p of pids) tries[p] = DEVICE_TRIES; // no use trying the rest
            }
        }
        const st = state();
        patchJob({ pdfs: st, step: st.waiting ? 'The server is fetching PDFs…' : '' });
        await updatePack(pk => allPapers(pk).forEach(p => { if (res.pdfs[p.id] !== undefined) p.offline_pdf = res.pdfs[p.id]; }));
        if (!st.waiting && !pids.some(due)) return;
        await sleep(PDF_POLL);
        if (run !== pdfRun) return;
    }
}

async function removePack() {
    if (!hasCaches()) return;
    await caches.delete(PACK);
    packStore.set(null);
    job.set(null);
}

// -----------------------------------------------------------------------------
// the page

const ago = t => {
    const s = Date.now() / 1000 - t;
    return s < 3600 ? Math.max(1, Math.round(s / 60)) + ' min ago' : s < 86400 * 2 ? Math.round(s / 3600) + ' h ago' : Math.round(s / 86400) + ' days ago';
};
const mb = n => (n / 1e6 < 10 ? (n / 1e6).toFixed(1) : Math.round(n / 1e6)) + ' MB';

const Progress = ({ j }) => {
    if (!j) return null;
    const p = j.pdfs;
    return html`<div class="pack-progress" role="status">
        ${j.step ? html`<div><span class="pack-dot" aria-hidden="true"></span>${j.step}</div>` : null}
        ${j.summaries ? html`<div>AI summaries: ${j.summaries[0]} of ${j.summaries[1]} ready</div>` : null}
        ${p ? html`<div>PDFs on this device: ${p.device} of ${p.total}
            ${p.waiting ? html`, ${p.waiting} still being fetched (about ${Math.max(1, Math.round(p.wait / 60))} min; arXiv wants one request every 15 seconds)` : ''}
            ${p.failed ? html`, ${p.failed} not available` : ''}
            ${p.stuck ? html`, ${p.stuck} couldn't be saved here (press the button to try again)` : ''}</div>` : null}
        ${j.pdfError ? html`<p class="form-error">${j.pdfError}</p>` : null}
        ${p && p.waiting ? html`<div class="muted">Keep this page open, or come back later: whatever is ready is downloaded when you open it.</div>` : null}
        ${j.note ? html`<div class="muted">${j.note}</div>` : null}
        ${j.error ? html`<p class="form-error">${j.error}</p>` : null}
    </div>`;
};

const Prepare = ({ pack, onDevice }) => {
    const j = job.use();
    const o = opts.use();
    const [usage, setUsage] = useState(null);
    useEffect(() => {
        if (navigator.storage && navigator.storage.estimate) navigator.storage.estimate().then(e => setUsage(e.usage)).catch(() => {});
    }, [pack, onDevice.size]);
    const running = j && j.running;
    return html`<section class="card pack-prepare">
        <h2>Reading offline</h2>
        <p>Before a flight, get your papers onto this device: what is in To read, your latest digest and the best of For you,
            with their AI summaries and PDFs. They open here without a network; votes, read marks and notes you make
            offline are sent once you are back online.</p>
        ${pack ? html`<p class="pack-status">On this device: ${allPapers(pack).length} papers from ${ago(pack.made)},
            ${onDevice.size} PDFs${usage ? html` (${mb(usage)} in all)` : null}.</p>` : null}
        ${pack && pack.pdfs_on === false ? null : html`
        <fieldset class="pack-pdfs" title="the PDFs of these lists are downloaded to this device">
            <legend>PDFs for</legend>
            ${PDF_SECTIONS.map(([k, label]) => html`<label key=${k} class="check"><input type="checkbox" checked=${o[k]}
                onChange=${e => setOpt(k, e.target.checked)} /> ${label}</label>`)}
            <span class="hint">at most ${pack ? pack.pdf_max : 60}; arXiv is asked for one every 15 s, so many take a while</span>
        </fieldset>`}
        <div class="pack-actions">
            <button class="btn btn-primary" disabled=${running} onClick=${getReady}
                    title="fetch the papers, their summaries and PDFs onto this device">${pack ? 'Update for offline' : 'Get ready for offline'}</button>
            ${pack ? html`<button class="btn" disabled=${running} onClick=${() => confirm('Remove the papers and PDFs from this device?') && removePack()}
                    title="frees the space on this device; nothing on your account changes">Remove from this device</button>` : null}
        </div>
        <${Progress} j=${j} />
        <p class="hint">Tip: add this site to your home screen, so the browser keeps the pack (Safari deletes a site's data
            after a week unused otherwise). The server keeps the PDFs ${pack ? pack.keep_days : 7} days, for you only.</p>
    </section>`;
};

const PackSection = ({ s, onDevice }) => {
    const [open, setOpen] = useState(true);
    return html`<section class="pack-section">
        <h2 class="pack-heading"><button class="linkbtn" aria-expanded=${open} onClick=${() => setOpen(!open)}>
            ${open ? '▾' : '▸'} ${s.title}</button> <span class="muted">${s.papers.length}</span></h2>
        ${open ? html`<div class="rel_papers">${s.papers.map(p => html`<div class="pack-paper" key=${p.id}>
            ${onDevice.has(p.id) ? html`<a class="btn btn-small btn-primary pack-read" href=${pdfPath(p.id)}
                    title="the PDF, from this device">📄 Read the PDF</a>`
                : p.offline_pdf === 'waiting' ? html`<span class="pack-pdf muted">PDF on its way to this device…</span>`
                : p.offline_pdf === 'failed' ? html`<span class="pack-pdf muted">no PDF for offline (the server couldn't get it)</span>` : null}
            <${Paper} paper=${onDevice.has(p.id) ? { ...p, pdf: pdfPath(p.id) } : p} quickTag=${s.tag || undefined} />
        </div>`)}</div>` : null}
    </section>`;
};

export const OfflinePage = () => {
    const m = me.use();
    const isOffline = offline.use();
    const pack = packStore.use();
    const j = job.use();
    const n = unsynced.use();
    const [onDevice, setOnDevice] = useState(new Set());
    useEffect(() => {
        if (pack === undefined) readPack().then(p => packStore.set(p));
    }, []);
    useEffect(() => {
        if (pack === undefined) return;
        pdfsOnDevice().then(setOnDevice);
        pageReady();
        // PDFs that were still being fetched: downloaded now
        // (and ready ones that didn't make it to the device yet)
        if (pack && !isOffline && !(j && j.running) && !(j && j.resumed) && m.user) {
            patchJob({ resumed: true });
            pdfsOnDevice().then(have => {
                const left = allPapers(pack)
                    .filter(p => p.offline_pdf === 'waiting' || (p.offline_pdf === 'ready' && !have.has(p.id)))
                    .map(p => p.id);
                if (left.length) return downloadPdfs(left);
            }).catch(e => patchJob({ pdfError: e.message }));
        }
    }, [pack, j && j.pdfs && j.pdfs.device]);

    if (!hasCaches()) {
        return html`<div class="page"><section class="card"><h2>Reading offline</h2>
            <p>This browser can't keep pages for offline use (it needs https and a current browser).</p></section></div>`;
    }
    if (!m.user && !isOffline) {
        return html`<div class="page"><section class="card"><h2>Reading offline</h2>
            <p><a href="/profile">Log in</a> to keep your papers on this device for reading offline.</p></section></div>`;
    }
    const page = { rank: 'offline', tags: '', viewing: false, owner: '', canEdit: canEdit(m), userTags: (pack && pack.tags) || [] };
    return html`<div class="page pack-page">
        ${isOffline ? html`<section class="card pack-offline" role="status">
            <b>You're offline.</b> ${pack ? html`These are the papers on this device, from ${ago(pack.made)}.` : "There are no papers on this device: get them ready next time you're online."}
            ${n ? html` ${n} change${n > 1 ? 's' : ''} (votes, read marks, notes) will be sent when you're back online.` : null}
            ${' '}<button class="linkbtn" onClick=${() => refreshMe().then(() => { offline.set(false); flushChanges(); }).catch(() => {})}
                    title="look whether the server can be reached again">Try to connect</button>
        </section>` : html`<${Prepare} pack=${pack} onDevice=${onDevice} />`}
        ${pack ? html`<${Page.Provider} value=${page}>
            ${pack.sections.map(s => html`<${PackSection} key=${s.key + s.title} s=${s} onDevice=${onDevice} />`)}
        </${Page.Provider}>` : null}
    </div>`;
};
