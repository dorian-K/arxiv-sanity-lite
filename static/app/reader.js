// the PDF reader of the reading pack (/read?pid=): the PDF on this device, drawn with the vendored
// pdf.js (loaded only here), so its links are ours. a tap on a citation shows the reference
// (the bibliography entry it points to, matched to the paper's references in the pack) with
// "+ Save" to a tag, offline too (offline.js saveRef: lib.js queues it); other links inside the
// paper jump, with a way back; a link to an arXiv paper can be saved the same way. pages are
// drawn when they come near the screen and let go of far from it (a phone has no memory for 60
// canvases). text can be selected, and the browser's find works on drawn pages
import { html, useState, useEffect, useRef, location, me, offline, canEdit, pageReady, api, CopyLink } from 'app/lib';
import { readPack, packPaper, pdfPath, refTag, refTagFor, saveRef } from 'app/offline';

const PDFJS = '/static/vendor/pdfjs-dist-4.10.38/'; // as in index.html's import map and update_vendor.py
const NEAR = '1500px'; // pages this close to the screen are drawn
const FAR = 4; // screens away from the visible part: a drawn page is let go of
const MAX_PIXELS = 16e6; // per canvas (browsers refuse bigger ones)
const ENTRY_CHARS = 350; // text taken from where a citation points: about one bibliography entry
const ZOOMS = [0.75, 1, 1.25, 1.5, 2, 3];

let lib = null;
async function pdfjs() {
    if (!lib) {
        lib = await import('pdfjs');
        lib.GlobalWorkerOptions.workerSrc = PDFJS + 'build/pdf.worker.min.mjs';
    }
    return lib;
}

// -----------------------------------------------------------------------------
// where a link goes, and what is written there

// {page (0-based), x, y} in PDF units, of a destination (a name or an explicit array)
async function resolveDest(pdf, dest) {
    const d = typeof dest === 'string' ? await pdf.getDestination(dest) : dest;
    if (!Array.isArray(d) || !d.length) return null;
    const page = typeof d[0] === 'object' && d[0] ? await pdf.getPageIndex(d[0]) : Number(d[0]);
    const kind = d[1] && d[1].name;
    const num = v => (typeof v === 'number' ? v : null);
    const [x, y] = kind === 'XYZ' ? [num(d[2]), num(d[3])] : kind === 'FitH' || kind === 'FitBH' ? [null, num(d[2])]
        : kind === 'FitR' ? [num(d[2]), num(d[5])] : [null, null];
    return { page, x, y };
}

// the text that starts where a destination points (a bibliography entry, for a citation): the
// topmost line just below the point, in its column, then on in reading order until the entry ends:
// with a hanging indent (or numbers in front), at the next line that starts as far left as the
// first; without one, at the first gap between lines wider than the narrowest
async function textAt(pdf, t) {
    const page = await pdf.getPage(t.page + 1);
    const items = (await page.getTextContent()).items.filter(i => typeof i.str === 'string');
    let start = -1, top = -Infinity, left = Infinity;
    if (t.y !== null) {
        items.forEach((it, i) => {
            const [x, y] = [it.transform[4], it.transform[5]];
            if (!it.str.trim() || y > t.y + 2 || y < t.y - 40) return;
            if (t.x && (x < t.x - 8 || x > t.x + 60)) return;
            if (y > top + 0.5 || (Math.abs(y - top) <= 0.5 && x < left)) { start = i; top = y; left = x; }
        });
    }
    if (start < 0) return '';
    const lines = []; // [x of its start, y, text]
    let chars = 0;
    for (let i = start; i < items.length && chars < 3 * ENTRY_CHARS && lines.length < 16; i++) {
        const it = items[i], [x, y] = [it.transform[4], it.transform[5]];
        const last = lines[lines.length - 1];
        if (!last || Math.abs(y - last[1]) > 1) {
            if (!it.str.trim()) continue;
            lines.push([x, y, '']);
        }
        lines[lines.length - 1][2] += it.str + (it.hasEOL ? ' ' : '');
        chars += it.str.length;
    }
    const x0 = lines[0][0];
    const hanging = lines.slice(1, 6).some(l => l[0] > x0 + 3);
    const gaps = lines.slice(1).map((l, i) => lines[i][1] - l[1]).filter(g => g > 0);
    const narrow = gaps.length ? Math.min(...gaps) : 0;
    let end = 1;
    for (; end < lines.length; end++) {
        const gap = lines[end - 1][1] - lines[end][1];
        if (gap <= 0) break; // another column or page
        if (hanging ? lines[end][0] <= x0 + 1.5 : gap > narrow * 1.3) break;
    }
    const s = lines.slice(0, end).map(l => l[2]).join(' ').slice(0, ENTRY_CHARS);
    // a hyphen at a line's end joins the word
    return s.replace(/\s+/g, ' ').replace(/(\w)- (\w)/g, '$1$2').trim();
}

const words = s => String(s || '').toLowerCase().normalize('NFKD').match(/[a-z0-9]{3,}/g) || [];

// the reference (of the pack's, for this paper) whose title is in the text: most of its words,
// the earliest (the text may run into the next entry)
export function matchRef(refs, text) {
    const tw = words(text);
    const pos = new Map();
    tw.forEach((w, i) => { if (!pos.has(w)) pos.set(w, i); });
    let best = null;
    for (const r of refs || []) {
        const rw = [...new Set(words(r.title))];
        if (rw.length < 2) continue;
        const hit = rw.filter(w => pos.has(w));
        const cover = hit.length / rw.length;
        if (cover < 0.7) continue;
        const score = cover - (0.5 * Math.min(...hit.map(w => pos.get(w)))) / (tw.length + 1);
        if (!best || score > best.score) best = { r, score };
    }
    return best ? best.r : null;
}

// an arXiv id in a link or a bibliography entry
const ARXIV = /(?:arxiv\.org\/(?:abs|pdf)\/|arxiv[:\s]+)(\d{4}\.\d{4,5})|\b(\d{4}\.\d{4,5})(?:v\d+)?\b/i;
const arxivId = s => { const m = String(s || '').match(ARXIV); return m ? m[1] || m[2] : null; };

// -----------------------------------------------------------------------------
// one page: a box of the right size, drawn when near the screen

const drawPage = async (el, page, scale, onLink, L, isLive) => {
    const vp = page.getViewport({ scale });
    const ratio = Math.min(window.devicePixelRatio || 1, Math.sqrt(MAX_PIXELS / (vp.width * vp.height)));
    const canvas = document.createElement('canvas');
    canvas.width = Math.floor(vp.width * ratio);
    canvas.height = Math.floor(vp.height * ratio);
    canvas.style.width = vp.width + 'px';
    canvas.style.height = vp.height + 'px';
    const task = page.render({ canvasContext: canvas.getContext('2d'), viewport: vp, transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : null });
    const cancel = () => task.cancel();
    el.replaceChildren(canvas);
    try { await task.promise; } catch (e) { return cancel; } // cancelled
    if (!isLive()) return cancel;
    const text = document.createElement('div');
    text.className = 'textLayer';
    text.style.setProperty('--scale-factor', scale);
    el.append(text);
    await new L.TextLayer({ textContentSource: page.streamTextContent(), container: text, viewport: vp }).render().catch(() => {});
    for (const a of await page.getAnnotations({ intent: 'display' })) {
        if (a.subtype !== 'Link' || !(a.dest || a.url)) continue;
        const [x1, y1, x2, y2] = vp.convertToViewportRectangle(a.rect);
        const link = document.createElement('a');
        link.className = 'pdf-link';
        link.href = a.url || '#';
        Object.assign(link.style, { left: Math.min(x1, x2) + 'px', top: Math.min(y1, y2) + 'px', width: Math.abs(x2 - x1) + 'px', height: Math.abs(y2 - y1) + 'px' });
        link.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); onLink(a); });
        el.append(link);
    }
    return cancel;
};

const PdfPage = ({ doc, i, scale, onLink }) => {
    const ref = useRef(null);
    const vp = doc.sizes[i];
    useEffect(() => {
        const el = ref.current;
        let drawn = false, live = true, cancel = null;
        const draw = async () => {
            if (drawn) return;
            drawn = true;
            const page = await doc.pdf.getPage(i + 1);
            if (live && drawn) cancel = await drawPage(el, page, scale, onLink, doc.lib, () => live && drawn);
        };
        const drop = () => { if (!drawn) return; drawn = false; if (cancel) cancel(); cancel = null; el.replaceChildren(); };
        const near = new IntersectionObserver(es => es.forEach(e => e.isIntersecting && draw()), { rootMargin: NEAR });
        const far = new IntersectionObserver(es => es.forEach(e => !e.isIntersecting && drop()), { rootMargin: `${FAR * 100}% 0px` });
        near.observe(el);
        far.observe(el);
        return () => { live = false; near.disconnect(); far.disconnect(); if (cancel) cancel(); el.replaceChildren(); };
    }, [doc, i, scale]);
    return html`<div class="pdf-page" data-page=${i + 1} ref=${ref}
        style=${{ width: vp.width * scale + 'px', height: vp.height * scale + 'px' }}></div>`;
};

// -----------------------------------------------------------------------------
// the reference a citation points to

const CitePop = ({ pop, tags, defTag, editable, onGo, onClose }) => {
    const chosen = refTag.use();
    const isOffline = offline.use();
    const to = refTagFor(chosen, tags, defTag, null);
    const r = pop.ref;
    const [saved, setSaved] = useState((r && r.saved) || []);
    const [msg, setMsg] = useState('');
    useEffect(() => {
        const esc = e => e.key === 'Escape' && onClose();
        document.addEventListener('keydown', esc);
        return () => document.removeEventListener('keydown', esc);
    }, []);
    const save = async () => {
        const tag = to;
        setSaved(s => [...s, tag]);
        setMsg('');
        try { setMsg(await saveRef(r, tag)); } catch (e) { setSaved(s => s.filter(t => t !== tag)); setMsg(e.message); }
    };
    const shown = (r && r.title) || '';
    const sameText = shown && words(shown).length >= 0.8 * words(pop.text).length;
    return html`<div class="cite-pop card" role="dialog" aria-label="reference">
        <button class="linkbtn cite-close" onClick=${onClose} aria-label="close" title="close (Esc)">×</button>
        ${shown ? html`<div class=${sameText ? 'cite-text' : 'cite-title'}>${shown}</div>` : null}
        ${pop.text && !sameText ? html`<div class=${shown ? 'cite-text muted' : 'cite-text'}>${pop.text}</div>` : null}
        ${pop.url ? html`<div class="cite-text"><a href=${pop.url} target="_blank" rel="noopener noreferrer">${pop.url}</a></div>` : null}
        ${r && r.id ? html`<div class="cite-meta muted">${r.id}${saved.length ? html` <span class="pack-ref-saved" title="saved to these tags of yours">✓ ${saved.join(', ')}</span>` : null}</div>` : null}
        <div class="cite-actions">
            ${r && r.id && editable && to ? html`
            <label title="the tag the paper is saved to; it then shows in To read">Save to
                <select value=${to} onChange=${e => refTag.set(e.target.value)}>${tags.map(t => html`<option key=${t} value=${t}>${t}</option>`)}</select></label>
            ${saved.includes(to) ? null : html`<button class="btn btn-small btn-primary" onClick=${save}
                title=${r.here ? 'save it to ' + to : 'save it to ' + to + '; the site fetches the paper first'}>+ Save</button>`}` : null}
            ${pop.target ? html`<button class="btn btn-small" onClick=${onGo} title="go to the entry in the bibliography (↩ brings you back)">Go to the reference</button>` : null}
        </div>
        ${!r || !r.id ? html`<p class="hint">${pop.url ? 'Not a paper this site can save.' : 'No arXiv or proceedings id is known for this reference, so it can\'t be saved here.'}
            ${!isOffline && shown ? html` <a href=${'/?q=' + encodeURIComponent(shown.slice(0, 200))}>Search the site for it</a>` : null}</p>` : null}
        ${msg ? html`<p class="hint">${msg}</p>` : null}
    </div>`;
};

// -----------------------------------------------------------------------------
// the page

export const ReaderPage = () => {
    const url = location.use();
    const m = me.use();
    const pid = url.searchParams.get('pid') || '';
    const [doc, setDoc] = useState(null);
    const [error, setError] = useState('');
    const [paper, setPaper] = useState(null);
    const [pack, setPack] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [width, setWidth] = useState(0);
    const [pop, setPop] = useState(null);
    const [back, setBack] = useState([]); // where jumps came from (scroll positions)
    const box = useRef(null);
    const linkTo = useRef(null); // the pages' links call the current onLink (it sees the latest paper and doc)

    useEffect(() => {
        readPack().then(pk => {
            setPack(pk);
            const p = packPaper(pk, pid);
            if (p) setPaper(p);
            // a copied link opened on a device without the pack: the title and the PDF's own address
            else if (!offline.get()) api('/ui/inspect?pid=' + encodeURIComponent(pid)).then(d => d && d.paper && setPaper(d.paper)).catch(() => {});
        });
    }, [pid]);
    useEffect(() => {
        const measure = () => box.current && setWidth(box.current.clientWidth);
        measure();
        window.addEventListener('resize', measure);
        return () => window.removeEventListener('resize', measure);
    }, []);
    useEffect(() => {
        let live = true, task = null;
        setDoc(null);
        setError('');
        (async () => {
            const L = await pdfjs();
            const r = await fetch(pdfPath(pid));
            if (!r.ok) throw new Error(r.status === 404 ? "This PDF isn't on this device, and the server doesn't keep it for you (it keeps a PDF for a week after you last got ready for offline)." : `The PDF couldn't be opened (the server answered ${r.status}).`);
            const data = new Uint8Array(await r.arrayBuffer());
            task = L.getDocument({ data, isEvalSupported: false, standardFontDataUrl: PDFJS + 'standard_fonts/' });
            const pdf = await task.promise;
            const sizes = [];
            for (let i = 1; i <= pdf.numPages; i++) sizes.push((await pdf.getPage(i)).getViewport({ scale: 1 }));
            if (live) setDoc({ pdf, sizes, lib: L });
        })().catch(e => live && setError(e.message || String(e))).finally(() => live && pageReady());
        return () => { live = false; if (task) task.destroy(); };
    }, [pid]);

    const fit = doc && width ? Math.min(width, 1000) / Math.max(...doc.sizes.map(v => v.width)) : 1;
    const scale = fit * zoom;
    const tags = (pack && pack.tags) || [];
    const editable = canEdit(m);

    const scrollTo = async t => {
        const el = box.current && box.current.querySelector(`.pdf-page[data-page="${t.page + 1}"]`);
        if (!el) return;
        const h = doc.sizes[t.page].height;
        const y = el.getBoundingClientRect().top + window.scrollY + (t.y === null ? 0 : (h - t.y) * scale);
        const header = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 0;
        setBack(b => [...b, window.scrollY]);
        window.scrollTo(0, Math.max(0, y - header - 60));
    };
    const goBack = () => setBack(b => { if (b.length) window.scrollTo(0, b[b.length - 1]); return b.slice(0, -1); });

    const onLink = async a => {
        const refs = (paper && paper.refs) || [];
        if (a.url) {
            const id = arxivId(a.url);
            if (!id) { window.open(a.url, '_blank', 'noopener,noreferrer'); return; }
            setPop({ url: a.url, ref: refs.find(r => r.id === id) || { id, title: '', here: false, saved: [] } });
            return;
        }
        const target = await resolveDest(doc.pdf, a.dest).catch(() => null);
        if (!target) return;
        const cite = typeof a.dest === 'string' && /^cite\./i.test(a.dest);
        const text = await textAt(doc.pdf, target).catch(() => '');
        let ref = matchRef(refs, text);
        const id = arxivId(text);
        if (ref && !ref.id && id) ref = { ...ref, id, here: false }; // /import finds it if it is here
        if (!ref && cite && id) ref = { id, title: '', here: false, saved: [] };
        if (cite || ref) setPop({ text, ref, target });
        else scrollTo(target);
    };

    linkTo.current = onLink;
    const follow = useRef(a => linkTo.current(a)).current;
    const title = paper ? paper.title : pid;
    return html`<div class="reader" ref=${box}>
        <div class="reader-bar">
            <a class="btn btn-small" href="/offline" title="the papers on this device">← Papers</a>
            <span class="reader-title" title=${title}>${title}</span>
            ${back.length ? html`<button class="btn btn-small" onClick=${goBack} title="back to where you were before following a link">↩ Back</button>` : null}
            <button class="btn btn-small" disabled=${zoom <= ZOOMS[0]} onClick=${() => setZoom(ZOOMS[Math.max(0, ZOOMS.indexOf(zoom) - 1)])} aria-label="zoom out" title="smaller">−</button>
            <button class="btn btn-small" disabled=${zoom >= ZOOMS[ZOOMS.length - 1]} onClick=${() => setZoom(ZOOMS[Math.min(ZOOMS.length - 1, ZOOMS.indexOf(zoom) + 1)])} aria-label="zoom in" title="bigger">+</button>
            <${CopyLink} href=${'/read?pid=' + encodeURIComponent(pid)} label="Copy link" title="copy a link to this PDF in the reader; it opens for you only (the PDF is on your devices, not public)" />
            <a class="btn btn-small reader-raw" href=${pdfPath(pid)} target="_blank" rel="noopener" title="the same PDF in your browser's own PDF viewer (citations can't be saved there)">Browser viewer</a>
        </div>
        <p class="hint reader-hint">Tap a citation to see the reference and save it to a tag; works offline too.</p>
        ${error ? html`<section class="card"><p class="form-error">${error}</p>
            ${paper ? html`<p><a href=${paper.pdf}>The PDF on ${paper.source || 'its site'}</a> (needs a network).</p>` : null}</section>`
        : !doc ? html`<div class="pdf-pages" role="status" aria-busy="true">
            <div class="skel-caption">Opening the PDF…</div>
            <div class="pdf-page skel" style=${{ width: Math.min(width || 600, 1000) + 'px', height: Math.min(width || 600, 1000) * 1.3 + 'px' }}></div></div>`
        : html`<div class="pdf-pages">${doc.sizes.map((_, i) => html`<${PdfPage} key=${i} doc=${doc} i=${i} scale=${scale} onLink=${follow} />`)}</div>`}
        ${pop ? html`<${CitePop} key=${(pop.ref && pop.ref.id) || pop.text || pop.url} pop=${pop} tags=${tags} defTag=${paper && (paper.utags || [])[0]}
            editable=${editable} onClose=${() => setPop(null)} onGo=${() => { scrollTo(pop.target); setPop(null); }} />` : null}
    </div>`;
};
