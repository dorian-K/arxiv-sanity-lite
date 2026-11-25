// reading without a network (on a plane): a pack of the user's papers kept on the device, and
// the page that shows it. "Get ready" asks the server for the pack (/ui/offline: To read, the
// latest digest, the best of For you), waits for their AI summaries, and downloads the PDFs that
// enrich_daemon.py fetches for it (/offline/pdfs; arXiv gets one request per paper at its pace,
// so that takes minutes). everything goes into the Cache Storage PACK, which sw.js answers from
// when the server can't be reached. votes, read marks and notes made offline wait in lib.js's
// queue, and the pack on the device follows them (onPaperChange). the pack also has the rest of
// the saved papers (closed, no summaries, thumbnails or PDFs), so the search box finds them offline.
// tabs show one kind of list (the address /digest opens the digest), an order puts what can be read
// here first, and each paper's references can be saved to a tag, offline too (queued in lib.js).
// a PDF whose paper leaves the PDF set (read, out of the digest) but stays in the pack isn't deleted:
// it becomes a light copy of ~200 kB (shrink.js: worse images, the reader's own fonts for the text,
// rounded drawings, stored LZMA-packed; nothing left out), up to LIGHT_BUDGET for all of them, the
// oldest let go of first
import { html, useState, useEffect, store, api, tagRequest, CopyLink, me, offline, unsynced, flushChanges, refreshMe, onPaperChange, canEdit, pageReady, location } from 'app/lib';
import { Paper, Page } from 'app/papers';
import { matcher } from 'app/highlight';

const PACK = 'pack-v1'; // as in sw.js
const PACK_URL = '/ui/offline';
const OPTS_KEY = 'resanity-pack-pdfs';
const PDF_POLL = 10000; // ms between asking how far the PDFs are
const PDF_POLL_FOR = 2 * 3600 * 1000; // ms it keeps asking, then stops (the server may be down for days)
const PDF_TIMEOUT = 5 * 60000; // one PDF's download to the device
const DEVICE_TRIES = 3; // downloads of one PDF to the device before giving it up
const SUMMARY_ROUNDS = 60; // ~3 minutes of asking for AI summaries
const LIGHT = 'X-Resanity-Light'; // header of a light copy on the device: its size in bytes
const PACKED = 'X-Resanity-Packed'; // a light copy stored packed by shrink.js (sw.js unpacks it)
const LIGHT_BUDGET = 300e6; // bytes of light copies on the device, all together
const LIGHT_MAX_ONE = 300e3; // a PDF shrink.js can't make lighter (pdf-lib can't read it) is kept only up to this

const hasCaches = () => typeof caches !== 'undefined';
const sleep = ms => new Promise(r => setTimeout(r, ms));
export const pdfPath = pid => '/offline/pdf/' + pid;

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

// pid -> 'full' or 'light', for the PDFs on the device, in the order they were stored
export async function pdfsOnDevice() {
    const out = new Map();
    if (!hasCaches()) return out;
    const c = await caches.open(PACK);
    for (const k of await c.keys()) {
        const path = new URL(k.url).pathname;
        if (!path.startsWith('/offline/pdf/')) continue;
        const r = await c.match(k);
        out.set(path.slice(13), r && r.headers.get(LIGHT) ? 'light' : 'full');
    }
    return out;
}

const allPapers = pack => pack.sections.flatMap(s => s.papers);
const unique = papers => [...new Map(papers.map(p => [p.id, p])).values()];
// the papers the pack is for, each once (the digest shows papers of To read again); the extra
// sections are only there to be searched
const mainPapers = pack => unique(pack.sections.filter(s => !s.extra).flatMap(s => s.papers));

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
        // PDFs and thumbnails of the papers no longer in the pack (read, out of the digest) leave the device
        if (old) {
            const keep = new Set(allPapers(pack).map(p => p.id));
            for (const pid of (await pdfsOnDevice()).keys()) if (!keep.has(pid)) await c.delete(pdfPath(pid));
            const keepThumbs = new Set(thumbs.map(u => new URL(u, window.location.href).pathname));
            for (const req of await c.keys()) {
                const path = new URL(req.url).pathname;
                if (path.startsWith('/thumb/') && !keepThumbs.has(path)) await c.delete(req);
            }
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
    let missing = mainPapers(pack).filter(p => !p.ai).map(p => p.id);
    const total = mainPapers(pack).length;
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
    // the whole set, even none: the server lets go of the rest, and so does the device
    const res = await api('/offline/pdfs', { pids: pids.join(',') }, { keep: true });
    if (res.left_out) patchJob({ note: `${res.left_out} PDFs left out: at most ${pack.pdf_max} at a time.` });
    const keep = new Set(pids);
    const leaving = [...(await pdfsOnDevice())].filter(([pid, kind]) => kind === 'full' && !keep.has(pid)).map(([pid]) => pid);
    await lighten(leaving, new Set(allPapers(pack).map(p => p.id)));
    if (!pids.length) return;
    await downloadPdfs(pids);
}

// the full PDFs of papers that left the set: a light copy each if the paper is still in the pack
// (shrink.js; pdf-lib is loaded only now), else deleted; then the oldest light copies over LIGHT_BUDGET go
export async function lighten(pids, inPack) {
    const c = await caches.open(PACK);
    let shrinkPdf = null;
    for (const [i, pid] of pids.entries()) {
        const r = await c.match(pdfPath(pid));
        if (!r || !inPack.has(pid)) { await c.delete(pdfPath(pid)); continue; }
        const head = `Making light copies of the PDFs you no longer need in full: ${i + 1} of ${pids.length}`;
        patchJob({ step: head + '…' });
        const bytes = new Uint8Array(await r.arrayBuffer());
        let light = null, small = false;
        try {
            shrinkPdf = shrinkPdf || (await import('app/shrink')).shrinkPdf;
            const got = await shrinkPdf(bytes, { progress: t => patchJob({ step: `${head} (${t})…` }) });
            if (got === 'not smaller') small = true; // compact already: it is its own light copy, whatever its size
            else light = got;
        } catch (e) { /* kept as it is if it is small, else let go of */ }
        const out = light || bytes; // a light copy is kept even above shrink.js's TARGET: it never leaves anything out
        if (!light && !small && out.length > LIGHT_MAX_ONE) { await c.delete(pdfPath(pid)); continue; }
        try {
            await c.delete(pdfPath(pid)); // so the copy goes to the end of the order: the newest
            const headers = { 'Content-Type': 'application/pdf', 'Content-Length': String(out.length), [LIGHT]: String(out.length) };
            if (light) headers[PACKED] = '1';
            await c.put(pdfPath(pid), new Response(new Blob([out]), { headers }));
        } catch (e) { /* no room: it is gone, as it was before light copies */ }
    }
    let total = 0;
    const copies = [];
    for (const k of await c.keys()) {
        if (!new URL(k.url).pathname.startsWith('/offline/pdf/')) continue;
        const n = Number((await c.match(k)).headers.get(LIGHT) || 0);
        if (n) { copies.push([k, n]); total += n; }
    }
    for (const [k, n] of copies) { // oldest first
        if (total <= LIGHT_BUDGET) break;
        await c.delete(k);
        total -= n;
    }
}

// one PDF from the server into the pack, its progress told as it comes in; throws with what went wrong
async function savePdf(c, pid, size, say) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), PDF_TIMEOUT);
    try {
        // no-store: past sw.js, which would answer with the light copy on the device
        const r = await fetch(pdfPath(pid), { signal: ctl.signal, cache: 'no-store' });
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
    const until = Date.now() + PDF_POLL_FOR;
    for (;;) {
        const res = await api('/ui/offline/pdfs?pids=' + encodeURIComponent(pids.join(',')));
        if (run !== pdfRun) return;
        const have = await pdfsOnDevice();
        const count = st => pids.filter(p => res.pdfs[p] === st).length;
        const full = p => have.get(p) === 'full'; // a light copy is replaced by the full PDF
        const due = p => res.pdfs[p] === 'ready' && !full(p) && (tries[p] || 0) < DEVICE_TRIES;
        const state = () => ({
            device: pids.filter(full).length, total: pids.length, waiting: count('waiting'),
            failed: count('failed'), stuck: pids.filter(p => res.pdfs[p] === 'ready' && !full(p) && tries[p] >= DEVICE_TRIES).length,
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
                have.set(pid, 'full');
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
        if (Date.now() > until) {
            patchJob({ step: `The server hasn't got ${st.waiting} of the PDFs yet. “Update for offline” later asks again.` });
            return;
        }
        await sleep(PDF_POLL);
        if (run !== pdfRun) return;
    }
}

async function removePack() {
    if (!hasCaches()) return;
    pdfRun++; // the download loop stops
    // and the server stops fetching PDFs for a pack that is gone
    api('/offline/pdfs', { pids: '' }, { keep: true }).catch(() => {});
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
const kinds = (m, kind) => [...m.values()].filter(k => k === kind).length;
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
        ${pack ? html`<p class="pack-status">On this device: ${mainPapers(pack).length} papers from ${ago(pack.made)},
            ${kinds(onDevice, 'full')} PDFs${kinds(onDevice, 'light') ? `, ${kinds(onDevice, 'light')} light copies of PDFs you're done with` : ''}${usage ? html` (${mb(usage)} in all)` : null}.</p>` : null}
        ${pack && pack.pdfs_on === false ? null : html`
        <fieldset class="pack-pdfs" title="the PDFs of these lists are downloaded to this device">
            <legend>PDFs for</legend>
            ${PDF_SECTIONS.map(([k, label]) => html`<label key=${k} class="check"><input type="checkbox" checked=${o[k]}
                onChange=${e => setOpt(k, e.target.checked)} /> ${label}</label>`)}
            <span class="hint">at most ${pack ? pack.pdf_max : 60} at a time; papers you've read leave the device when you update, making room; arXiv is asked for one every 15 s, so many take a while</span>
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

// a card whose PDF is on the device: its title and pdf link open that copy, its site gets a link of its own
export const readerPath = pid => '/read?pid=' + encodeURIComponent(pid);
const onDeviceCard = p => ({ ...p, url: readerPath(p.id), pdf: pdfPath(p.id), source_url: p.url });

// what a paper cites, from the pack: the ones with a known id can be saved to a tag (then they are in
// To read), offline too: /add for papers the site has, /import (it fetches the paper) for the others
export const refTag = store(''); // the tag cited papers are saved to, as last chosen

// the tag a cited paper of p goes to: the one last chosen, else the list's, else one p is in
export const refTagFor = (chosen, tags, tag, p) =>
    tags.includes(chosen) ? chosen : (tag && tags.includes(tag) ? tag : '') || ((p && p.utags) || []).find(t => tags.includes(t)) || tags[0] || '';

// saves a cited paper ({id, here}) to a tag: /add if the site has it, else /import (which fetches it);
// offline both wait in lib.js's queue. resolves to a note to show ('' if none), throws with the reason
export async function saveRef(r, tag) {
    const res = r.here ? await tagRequest('/add/' + r.id + '/' + encodeURIComponent(tag))
        : await api('/import', { text: r.id, tag });
    const got = res && res.results && res.results[0];
    if (got && (!got.pid || !/already here|added from/.test(got.status))) throw new Error(got.status); // not found, over the limit
    await updatePack(pack => allPapers(pack).forEach(q => (q.refs || []).forEach(x => {
        if (x.id === r.id && !(x.saved || []).includes(tag)) x.saved = [...(x.saved || []), tag];
    })));
    return typeof res === 'string' && res.includes('kept on this device') ? 'sent when you are back online' : '';
}

// the paper in the pack, from its first list
export const packPaper = (pack, pid) => (pack ? allPapers(pack).find(p => p.id === pid) : null) || null;
const PackRefs = ({ p, tag, tags, canEdit }) => {
    const [open, setOpen] = useState(false);
    const chosen = refTag.use();
    const refs = p.refs || [];
    if (!refs.length) return null;
    const to = refTagFor(chosen, tags, tag, p);
    return html`<div class="pack-refs">
        <button class="linkbtn" aria-expanded=${open} onClick=${() => setOpen(!open)}
                title="the papers this one cites; save the ones you want to read to a tag, offline too">
            ${open ? '▾' : '▸'} References (${refs.length})</button>
        ${open ? html`
        ${canEdit && to ? html`<div class="pack-refs-to">
            <label title="the tag “+ Save” puts a cited paper in; it then shows in To read">Save to
                <select value=${to} onChange=${e => refTag.set(e.target.value)}>
                    ${tags.map(t => html`<option key=${t} value=${t}>${t}</option>`)}
                </select></label>
            <span class="hint">saved papers go to To read; offline, they are sent when you're back online</span>
        </div>` : null}
        <ol class="pack-refs-list">${refs.map((r, i) => html`<${PackRef} key=${i} r=${r} to=${to} canEdit=${canEdit && to} />`)}</ol>` : null}
    </div>`;
};

const PackRef = ({ r, to, canEdit }) => {
    const [saved, setSaved] = useState(r.saved || []);
    const [msg, setMsg] = useState('');
    const save = async () => {
        const tag = to;
        setSaved(s => [...s, tag]);
        setMsg('');
        try {
            setMsg(await saveRef(r, tag));
        } catch (e) {
            setSaved(s => s.filter(t => t !== tag));
            setMsg(e.message);
        }
    };
    const title = r.url ? html`<a href=${r.url}>${r.title}</a>` : r.title;
    return html`<li>
        ${title}${r.id ? html` <span class="muted">${r.id}</span>` : null}
        ${saved.length ? html` <span class="pack-ref-saved" title="saved to these tags of yours">✓ ${saved.join(', ')}</span>` : null}
        ${canEdit && r.id && !saved.includes(to) ? html` <button class="btn btn-small" onClick=${save}
                title=${'save it to ' + to + (r.here ? '' : '; the site fetches the paper first')}>+ Save</button>` : null}
        ${!r.id ? html` <span class="muted" title="no arXiv or proceedings id was found for this reference">(not on a site we know)</span>` : null}
        ${msg ? html` <span class="muted">${msg}</span>` : null}
    </li>`;
};

const PackPaper = ({ p, tag, onDevice, tags, canEdit }) => html`<div class="pack-paper">
    ${onDevice.get(p.id) === 'full' ? html`<a class="btn btn-small btn-primary pack-read" href=${readerPath(p.id)}
            title="the PDF, from this device; tap a citation in it to see the reference and save it to a tag">📄 Read the PDF</a>`
        : onDevice.get(p.id) === 'light' ? html`<a class="btn btn-small pack-read" href=${readerPath(p.id)}
            title="kept small to save space on this device: every page, with lower-quality images and standard fonts for the text">📄 Read the PDF (light copy)</a>`
        : null}
    ${onDevice.has(p.id) ? html`<span class="pack-read-more">
        <a href=${pdfPath(p.id)} target="_blank" rel="noopener" title="the same PDF in your browser's own PDF viewer (citations can't be saved there)">browser viewer</a>
        <${CopyLink} href=${readerPath(p.id)} label="copy link" className="linkbtn"
            title="copy a link to this PDF in the reader; it opens for you only (the PDF is on your devices, not public)" />
    </span>`
        : p.offline_pdf === 'waiting' ? html`<span class="pack-pdf muted">PDF on its way to this device…</span>`
        : p.offline_pdf === 'failed' ? html`<span class="pack-pdf muted">no PDF for offline (the server couldn't get it)</span>` : null}
    <${Paper} paper=${onDevice.has(p.id) ? onDeviceCard(p) : p} quickTag=${tag || undefined} />
    <${PackRefs} p=${p} tag=${tag} tags=${tags || []} canEdit=${canEdit} />
</div>`;

const PackSection = ({ s, onDevice, order, page, alone }) => {
    const [open, setOpen] = useState(!s.extra || alone);
    return html`<section class="pack-section">
        <h2 class="pack-heading"><button class="linkbtn" aria-expanded=${open} onClick=${() => setOpen(!open)}>
            ${open ? '▾' : '▸'} ${s.title}</button> <span class="muted">${s.papers.length}</span></h2>
        ${s.extra && !open ? html`<p class="hint pack-extra-hint">Kept for searching: no PDFs or thumbnails.</p>` : null}
        ${open ? html`<div class="rel_papers">${sorted(s.papers, order, onDevice).map(p => html`<${PackPaper} key=${p.id} p=${p} tag=${s.tag}
            onDevice=${onDevice} tags=${page.userTags} canEdit=${page.canEdit} />`)}</div>` : null}
    </section>`;
};

// the order within each list. 'here': what can be read on this device first (the full PDF, then a
// light copy, then the rest), otherwise as the site had it: To read and Saved newest first, the
// digest and For you best first
const ORDERS = [
    ['here', 'PDF on this device first', 'papers you can read here, without a network, come first; otherwise as on the site'],
    ['site', 'As on the site', 'To read and Saved newest first, the digest and For you best match first'],
    ['newest', 'Newest papers first', 'by the date of the paper'],
];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const paperDate = p => { // "Nov 25 2026"
    const m = String(p.time || '').match(/^(\w{3}) (\d+) (\d{4})/);
    return m ? Number(m[3]) * 10000 + (MONTHS.indexOf(m[1]) + 1) * 100 + Number(m[2]) : 0;
};
function sorted(papers, order, onDevice) {
    const key = order === 'here' ? p => ({ full: 0, light: 1 })[onDevice.get(p.id)] ?? 2
        : order === 'newest' ? p => -paperDate(p) : null;
    if (!key) return papers;
    return papers.map((p, i) => [key(p), i, p]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(x => x[2]);
}
const ORDER_KEY = 'resanity-pack-order';
const packOrder = store((() => { try { return localStorage.getItem(ORDER_KEY) || 'here'; } catch (e) { return 'here'; } })());
const setOrder = o => { packOrder.set(o); try { localStorage.setItem(ORDER_KEY, o); } catch (e) { /* not remembered */ } };

// the lists of the pack as tabs; the address picks the first one (/digest: the digest)
const VIEWS = [['all', 'Everything'], ['toread', 'To read'], ['digest', 'Digest'], ['foryou', 'For you'], ['saved', 'Saved']];
const VIEW_TIPS = {
    all: 'every list on this device', toread: 'saved papers not marked as read', digest: 'your latest digest, as it was sent',
    foryou: 'the best new papers for your tags', saved: 'your other saved papers, read ones too: no PDFs or thumbnails',
};
const viewOf = path => (path === '/digest' ? 'digest' : 'all');

const PackLists = ({ pack, onDevice, page }) => {
    const url = location.use();
    const [view, setView] = useState(viewOf(url.pathname));
    useEffect(() => setView(viewOf(url.pathname)), [url.pathname]);
    const order = packOrder.use();
    const count = key => unique(pack.sections.filter(s => s.key === key).flatMap(s => s.papers)).length;
    const shown = pack.sections.filter(s => view === 'all' || s.key === view);
    return html`<div class="pack-lists">
        <div class="tabrow">
            <nav class="tabs pack-tabs" aria-label="lists on this device">
                ${VIEWS.filter(([k]) => k === 'all' || count(k)).map(([k, label]) => html`<button key=${k} type="button"
                    aria-pressed=${view === k} class=${view === k ? 'active' : ''} title=${VIEW_TIPS[k]}
                    onClick=${() => setView(k)}>${label}${k === 'all' ? '' : ` (${count(k)})`}</button>`)}
            </nav>
            <label class="window" title=${ORDERS.find(o => o[0] === order)?.[2]}>
                <select aria-label="order" value=${order} onChange=${e => setOrder(e.target.value)}>
                    ${ORDERS.map(([k, label]) => html`<option key=${k} value=${k}>${label}</option>`)}
                </select>
            </label>
        </div>
        <p class="hint pack-order-hint">${ORDERS.find(o => o[0] === order)?.[2]}${view === 'digest' && pack.digest_time
            ? `. Your digest of ${pack.digest_time}: 👍 👎 😐 work offline too and are sent when you're back online.` : '.'}</p>
        ${view === 'digest' && !count('digest') ? html`<p class="muted">There was no digest when this device got its papers.</p>` : null}
        ${shown.map(s => html`<${PackSection} key=${view + s.key + s.title} s=${s} onDevice=${onDevice} order=${order} page=${page}
            alone=${view !== 'all'} />`)}
    </div>`;
};

// searching the pack, without a network: like the site's search, every word the query has (at a
// word's start, stopwords left out), best where it is in the title, the note or the authors
const SEARCH_SHOW = 50;
const WORD_RE = /[\p{L}\p{N}]+/gu;
const FIELDS = [
    [p => p.title, 4], [p => p.note, 4], [p => p.authors, 3], [p => p.id, 3],
    [p => (p.ai ? [p.ai.problem, p.ai.approach, p.ai.result, p.ai.caveat].join(' ') : ''), 1], [p => p.summary, 1],
];
const esc = w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const wordRe = w => new RegExp('(?<![\\p{L}\\p{N}])' + esc(w) + (w.length < 3 ? '(?![\\p{L}\\p{N}])' : ''), 'iu');

export function searchPack(pack, q) {
    const stop = new Set(pack.stopwords || []);
    const words = (String(q).toLowerCase().match(WORD_RE) || []).slice(0, 8);
    const terms = words.map(w => [w, stop.has(w)]);
    let need = [...new Set(words.filter(w => !stop.has(w)))];
    if (!need.length) need = [...new Set(words)];
    if (!need.length) return { papers: [], terms };
    const res = need.map(wordRe);
    const seen = new Set(), found = [];
    pack.sections.forEach(s => s.papers.forEach((p, i) => {
        if (seen.has(p.id)) return;
        seen.add(p.id);
        const texts = FIELDS.map(([f, w]) => [String(f(p) || ''), w]);
        let score = 0;
        for (const re of res) {
            const hits = texts.filter(([t]) => re.test(t));
            if (!hits.length) return;
            score += Math.max(...hits.map(([, w]) => w));
        }
        found.push({ p, tag: s.tag, score, order: found.length });
    }));
    found.sort((a, b) => b.score - a.score || a.order - b.order);
    return { papers: found, terms };
}

const PackSearch = ({ pack, onDevice, page, children }) => {
    const [q, setQ] = useState('');
    const [all, setAll] = useState(false);
    const res = q.trim() ? searchPack(pack, q) : null;
    const shown = res ? (all ? res.papers : res.papers.slice(0, SEARCH_SHOW)) : [];
    return html`<div class="pack-search">
        <input class="form-control" type="search" value=${q} placeholder="Search the papers on this device"
            aria-label="search the papers on this device" title="titles, authors, abstracts, AI summaries and your notes; works offline"
            onInput=${e => { setQ(e.target.value); setAll(false); }} onKeyDown=${e => { if (e.key === 'Escape') setQ(''); }} />
        ${res ? html`<p class="muted pack-search-count">${res.papers.length ? `${res.papers.length} of the ${new Set(allPapers(pack).map(p => p.id)).size} papers on this device match` : 'No paper on this device matches.'}</p>
            <${Page.Provider} value=${{ ...page, hl: matcher(res.terms) }}>
                <div class="rel_papers">${shown.map(r => html`<${PackPaper} key=${r.p.id} p=${r.p} tag=${r.tag} onDevice=${onDevice}
                    tags=${page.userTags} canEdit=${page.canEdit} />`)}</div>
            </${Page.Provider}>
            ${res.papers.length > shown.length ? html`<button class="btn" onClick=${() => setAll(true)}>Show all ${res.papers.length}</button>` : null}` : children}
    </div>`;
};

export const OfflinePage = () => {
    const m = me.use();
    const isOffline = offline.use();
    const pack = packStore.use();
    const j = job.use();
    const n = unsynced.use();
    const [onDevice, setOnDevice] = useState(new Map());
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
                    .filter(p => p.offline_pdf === 'waiting' || (p.offline_pdf === 'ready' && have.get(p.id) !== 'full'))
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
        ${pack ? html`<${PackSearch} pack=${pack} onDevice=${onDevice} page=${page}>
            <${Page.Provider} value=${page}>
                <${PackLists} pack=${pack} onDevice=${onDevice} page=${page} />
            </${Page.Provider}>
        </${PackSearch}>` : null}
    </div>`;
};
