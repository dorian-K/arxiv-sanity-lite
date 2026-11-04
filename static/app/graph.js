// the citation graph (/graph?pid=), drawn as a river of time: every paper sits at its year, oldest
// on the left, so a paper's references lie to its left and the papers citing it to its right, and
// an area's foundations collect on the left as it is explored. Papers many others here cite grow.
// Hovering traces a paper's lineage (everything it builds on and everything building on it).
// Clicking a paper whose references are known spreads them out; one that hasn't been read is asked
// for (/graph/analyze) and spreads out once the server has read it. Height is topic: the main
// direction in which the embeddings of the paper's neighbourhood differ, so threads of related work
// form bands. Faint "ghosts" are papers closest to it in content that no citation links to it:
// related work it missed, or that missed it. A force layout in SVG, no library
import { html, useState, useEffect, useRef, me, api, location, Err } from 'app/lib';
import { MathText } from 'app/math';

const YEAR_GAP = 120; // graph units between two years that have papers
const TOPIC_GAP = 90; // graph units per standard deviation on the topic axis
const POLL = 6000; // ms between looks at papers waiting to be read
const SECONDS_PER_PAPER = 60; // enrich_daemon.py: up to three requests to arxiv.org, 20 s apart

// the graph's state outside Preact: positions change every frame
class Graph {
    constructor() {
        this.nodes = new Map(); // id -> {id, title, ..., x, y, vx, vy, expanded, born}
        this.edges = new Map(); // 'a>b' -> [a, b]: a cites b
        this.alpha = 1;
        this.years = [];
    }

    // a response of /ui/graph: new papers sprout from the one they came from
    merge(res, from) {
        const origin = this.nodes.get(from) || { x: 0, y: 0 };
        const add = (n, ghost) => {
            const old = this.nodes.get(n.id);
            if (old) {
                if (ghost) return; // already on the graph for real
                Object.assign(old, n, { expanded: old.expanded, ghost: false });
            } else {
                const y = n.topic !== undefined ? n.topic * TOPIC_GAP : origin.y;
                this.nodes.set(n.id, { ...n, ghost, x: origin.x + (Math.random() - 0.5) * 20, y: y + (Math.random() - 0.5) * 30, vx: 0, vy: 0, born: performance.now() });
            }
        };
        for (const n of res.nodes) add(n, false);
        for (const n of res.ghosts || []) add(n, true);
        for (const [a, b] of res.edges) if (!this.edges.has(a + '>' + b)) this.edges.set(a + '>' + b, [a, b]);
        const self = this.nodes.get(res.pid);
        self.expanded = true;
        self.ghost = false;
        this.years = [...new Set([...this.nodes.values()].map(n => n.year).filter(Boolean))].sort();
        this.alpha = Math.max(this.alpha, 0.9);
    }

    // where a year sits: years with papers side by side, gaps in time don't take room
    yearX(year) {
        const i = this.years.indexOf(year);
        return (i < 0 ? this.years.length / 2 : i) * YEAR_GAP - ((this.years.length - 1) * YEAR_GAP) / 2;
    }

    // how often each paper is cited here, and who cites whom
    stats() {
        const cited = new Map(), out = new Map(), inn = new Map();
        for (const [a, b] of this.edges.values()) {
            cited.set(b, (cited.get(b) || 0) + 1);
            if (!out.has(a)) out.set(a, []);
            out.get(a).push(b);
            if (!inn.has(b)) inn.set(b, []);
            inn.get(b).push(a);
        }
        return { cited, out, inn };
    }

    // one step of the layout: each paper is pulled to its year, papers push each other apart
    // vertically, and citations pull related papers to similar heights
    tick(dragging, cited) {
        if (this.alpha < 0.005) return false;
        const ns = [...this.nodes.values()];
        const k = this.alpha;
        for (let i = 0; i < ns.length; i++) {
            const a = ns[i];
            for (let j = i + 1; j < ns.length; j++) {
                const b = ns[j];
                const dx = a.x - b.x;
                if (Math.abs(dx) > YEAR_GAP) continue;
                let dy = a.y - b.y;
                if (Math.abs(dy) < 0.5) dy = Math.random() - 0.5;
                const d2 = dx * dx + dy * dy;
                if (d2 > 40000) continue;
                const f = (700 * k) / Math.max(d2, 30);
                a.vx += dx * f; a.vy += dy * f;
                b.vx -= dx * f; b.vy -= dy * f;
            }
        }
        for (const [ia, ib] of this.edges.values()) {
            const a = this.nodes.get(ia), b = this.nodes.get(ib);
            const dy = b.y - a.y;
            a.vy += dy * 0.012 * k; b.vy -= dy * 0.012 * k;
        }
        for (const n of ns) {
            const yx = this.yearX(n.year);
            n.vx += (yx - n.x) * 0.02 * k;
            if (n.topic !== undefined) n.vy += (n.topic * TOPIC_GAP - n.y) * 0.03 * k; // its topic's height
            else n.vy -= n.y * 0.004 * k;
            if (n === dragging) { n.vx = 0; n.vy = 0; continue; }
            n.vx *= 0.6; n.vy *= 0.6;
            n.x += n.vx; n.y += n.vy;
            n.x = Math.max(yx - YEAR_GAP * 0.42, Math.min(yx + YEAR_GAP * 0.42, n.x)); // within its year
        }
        this.alpha *= 0.985;
        return true;
    }

    bounds() {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const n of this.nodes.values()) {
            x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y);
        }
        return x0 === Infinity ? [-100, -100, 100, 100] : [x0, y0, x1, y1];
    }
}

// everything a paper builds on (following its references) or that builds on it, as far as known
const reach = (start, next) => {
    const seen = new Set([start]), todo = [start];
    while (todo.length) for (const x of next.get(todo.pop()) || []) if (!seen.has(x)) { seen.add(x); todo.push(x); }
    return seen;
};

const radius = (id, cited, center) => (center ? 13 : 6 + 3.2 * Math.sqrt(cited.get(id) || 0));
// the years on the ruler, in screen x, leaving out those that would overlap the one before
const rulerYears = (g, v) => {
    const out = [];
    for (const y of g.years) {
        const x = g.yearX(y) * v.k + v.x;
        if (!out.length || x - out[out.length - 1][1] >= 40) out.push([y, x]);
    }
    return out;
};
const short = (t, n = 38) => (t.length > n ? t.slice(0, n - 1) + '…' : t);
const STATE_TEXT = {
    analyzed: 'References read',
    queued: 'Being read…',
    failed: "References couldn't be read",
    new: 'References not read yet',
};

export const GraphPage = () => {
    const url = location.use();
    const pid = url.searchParams.get('pid') || '';
    const m = me.use();
    const gRef = useRef(null);
    const svgRef = useRef(null);
    const [, setFrame] = useState(0);
    const [res, setRes] = useState(null); // the first answer: the center, the queue, whether reading is on
    const [error, setError] = useState('');
    const [selected, setSelected] = useState(pid);
    const [hover, setHover] = useState(null);
    const [busy, setBusy] = useState(false);
    const [actionError, setActionError] = useState('');
    const view = useRef({ x: 0, y: 0, k: 1 }); // pan and zoom
    const drag = useRef(null);
    const pointers = useRef(new Map());
    const fitted = useRef(false); // fit the view to the graph until the user moves it
    const redraw = () => setFrame(f => f + 1);

    const fit = () => {
        const svg = svgRef.current, g = gRef.current;
        if (!svg || !g) return;
        const [x0, y0, x1, y1] = g.bounds();
        const w = svg.clientWidth, h = svg.clientHeight;
        const k = Math.min(1.6, 0.92 * Math.min(w / (x1 - x0 + 180), (h - 40) / (y1 - y0 + 90)));
        view.current = { k, x: w / 2 - ((x0 + x1) / 2) * k, y: 20 + (h - 20) / 2 - ((y0 + y1) / 2) * k };
        redraw();
    };

    // the layout runs while it moves, and stops when it has settled
    const running = useRef(false);
    const run = () => {
        if (running.current) return;
        running.current = true;
        const step = () => {
            const g = gRef.current;
            if (!g) { running.current = false; return; }
            const moved = g.tick(drag.current && drag.current.node, g.stats().cited);
            if (fitted.current) fit();
            else redraw();
            if (moved || drag.current) requestAnimationFrame(step);
            else running.current = false;
        };
        requestAnimationFrame(step);
    };

    // the topic axis is the first paper's (root), so papers added later land on the same scale
    const fetchAround = id => api('/ui/graph?pid=' + encodeURIComponent(id) + '&root=' + encodeURIComponent(pid));

    const expand = id => {
        setBusy(true); setActionError('');
        return fetchAround(id).then(r => {
            gRef.current.merge(r, id);
            setRes(x => ({ ...x, queue: r.queue }));
            run();
        }).catch(e => setActionError(e.message)).finally(() => setBusy(false));
    };

    const analyze = id => {
        setBusy(true); setActionError('');
        api('/graph/analyze', { pid: id }).then(r => {
            const n = gRef.current && gRef.current.nodes.get(id);
            if (!n) return;
            n.state = r.state;
            if (r.state === 'analyzed') expand(id);
            redraw();
        }).catch(e => setActionError(e.message)).finally(() => setBusy(false));
    };

    // a new center: a new graph
    useEffect(() => {
        gRef.current = new Graph();
        fitted.current = true;
        setRes(null); setError(''); setSelected(pid); setActionError('');
        fetchAround(pid).then(r => {
            const g = gRef.current;
            g.merge(r, pid);
            setRes(r);
            const c = g.nodes.get(pid);
            c.x = g.yearX(c.year); c.y = 0;
            // the paper this graph is about: read it right away if it hasn't been yet
            if (c.state === 'new' && r.enrich_on && me.get().logged_in) analyze(pid);
            run();
        }).catch(e => setError(e.message));
        return () => { gRef.current = null; };
    }, [pid]);

    // papers waiting to be read: looked at every few seconds, and spread out once they are
    useEffect(() => {
        const timer = setInterval(() => {
            const g = gRef.current;
            if (!g) return;
            for (const n of [...g.nodes.values()].filter(x => x.state === 'queued').slice(0, 5)) {
                fetchAround(n.id).then(r => {
                    const now = r.nodes.find(x => x.id === n.id);
                    setRes(x => (x ? { ...x, queue: r.queue } : x));
                    if (now && now.state !== 'queued') {
                        if (now.state === 'analyzed') gRef.current.merge(r, n.id);
                        else Object.assign(n, now);
                        run();
                    }
                }).catch(() => {});
            }
        }, POLL);
        return () => clearInterval(timer);
    }, [pid]);

    // a click on a paper selects it, spreads out its citations if they are known, or asks for them
    const clickNode = n => {
        setSelected(n.id);
        if (n.ghost) { expand(n.id); return; } // onto the graph for real, with its citations
        if (n.state === 'analyzed' && !n.expanded) expand(n.id);
        else if (n.state === 'new' && res && res.enrich_on && m.logged_in) analyze(n.id);
        else redraw();
    };

    // pointer handling: drag a paper, drag the background to pan, wheel or pinch to zoom
    const toGraph = e => {
        const r = svgRef.current.getBoundingClientRect(), v = view.current;
        return [(e.clientX - r.left - v.x) / v.k, (e.clientY - r.top - v.y) / v.k];
    };
    const zoomAt = (cx, cy, factor) => {
        const v = view.current;
        const k = Math.min(6, Math.max(0.1, v.k * factor));
        view.current = { k, x: cx - ((cx - v.x) * k) / v.k, y: cy - ((cy - v.y) * k) / v.k };
        fitted.current = false;
        redraw();
    };
    useEffect(() => {
        const svg = svgRef.current;
        if (!svg) return undefined;
        const wheel = e => {
            e.preventDefault();
            const r = svg.getBoundingClientRect();
            zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
        };
        svg.addEventListener('wheel', wheel, { passive: false });
        return () => svg.removeEventListener('wheel', wheel);
    }, [res !== null]);

    const down = (e, node) => {
        e.stopPropagation();
        try { svgRef.current.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is gone already */ }
        pointers.current.set(e.pointerId, [e.clientX, e.clientY]);
        if (pointers.current.size === 2) { drag.current = { pinch: true }; return; }
        drag.current = { node, x: e.clientX, y: e.clientY, moved: false, vx: view.current.x, vy: view.current.y };
    };
    const move = e => {
        const d = drag.current;
        if (!d) return;
        const before = pointers.current.get(e.pointerId);
        pointers.current.set(e.pointerId, [e.clientX, e.clientY]);
        if (d.pinch && pointers.current.size === 2 && before) {
            const [a, b] = [...pointers.current.values()];
            const other = [...pointers.current.entries()].find(([id]) => id !== e.pointerId)[1];
            const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);
            const r = svgRef.current.getBoundingClientRect();
            zoomAt((a[0] + b[0]) / 2 - r.left, (a[1] + b[1]) / 2 - r.top, dist([e.clientX, e.clientY], other) / (dist(before, other) || 1));
            return;
        }
        if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 4) d.moved = true;
        if (!d.moved) return;
        fitted.current = false;
        if (d.node) {
            const [x, y] = toGraph(e);
            d.node.x = x; d.node.y = y;
            gRef.current.alpha = Math.max(gRef.current.alpha, 0.3);
            run();
        } else {
            view.current = { ...view.current, x: d.vx + e.clientX - d.x, y: d.vy + e.clientY - d.y };
            redraw();
        }
    };
    const up = e => {
        pointers.current.delete(e.pointerId);
        const d = drag.current;
        drag.current = null;
        if (d && !d.moved && !d.pinch && d.node) clickNode(d.node);
        else if (d && !d.moved && !d.pinch && !d.node) setSelected(null);
    };

    if (error) return html`<div class="page"><section class="card"><p class="form-error">${error}</p></section></div>`;
    const g = gRef.current;
    const v = view.current;
    const center = g && g.nodes.get(pid);
    const { cited, out, inn } = g ? g.stats() : { cited: new Map(), out: new Map(), inn: new Map() };
    // the lineage of the hovered (else the selected) paper: what it builds on, what builds on it
    const focus = hover || selected;
    const up_ = focus && g ? reach(focus, out) : null;
    const down_ = focus && g ? reach(focus, inn) : null;
    const lit = id => !focus || up_.has(id) || down_.has(id);
    // which side of the center a paper is on, for its color
    const ancestors = g ? reach(pid, out) : new Set(), descendants = g ? reach(pid, inn) : new Set();
    const side = id => (id === pid ? 'center' : ancestors.has(id) ? 'past' : descendants.has(id) ? 'future' : 'other');
    const all = g ? [...g.nodes.values()] : [];
    const nodes = all.filter(n => !n.ghost);
    const ghostList = all.filter(n => n.ghost).sort((a, b) => b.similarity - a.similarity);
    const top = nodes.filter(n => (cited.get(n.id) || 0) >= 2 && n.id !== pid).sort((a, b) => cited.get(b.id) - cited.get(a.id));
    // labels: the paper itself, the hovered and selected ones, then the most cited, then (zoomed in)
    // any; one is left out where it would cover a label already placed, so they never pile up
    const want = [pid, hover, selected, ...top.map(n => n.id), ...(v.k > 1.2 ? nodes.map(n => n.id) : [])];
    const labeled = new Set(), boxes = [];
    for (const id of want) {
        const n = id && g && g.nodes.get(id);
        if (!n || labeled.has(id)) continue;
        const w = Math.min(n.title.length || 10, id === pid ? 60 : 38) * 6.6, sx = n.x * v.k + v.x, sy = n.y * v.k + v.y - 18;
        const box = [sx - w / 2, sy - 8, sx + w / 2, sy + 8];
        const free = !boxes.some(b => b[0] < box[2] && box[0] < b[2] && b[1] < box[3] && box[1] < b[3]);
        if (free || id === hover || id === selected) { labeled.add(id); boxes.push(box); }
        if (labeled.size > 40) break;
    }
    const sel = g && selected && g.nodes.get(selected);
    const hov = g && hover && g.nodes.get(hover);
    const queue = (res && res.queue) || [];
    const reading = nodes.filter(n => n.state === 'queued').length;
    const now = performance.now();
    // papers keep a readable size when the view is zoomed out to fit a wide graph
    const grow = v.k < 1 ? Math.pow(1 / v.k, 0.65) : 1;
    const h = svgRef.current ? svgRef.current.clientHeight : 600;

    return html`
    <div class="graph-page">
        <section class="graph-head">
            <div class="graph-kicker">Citation graph</div>
            <h1>${center ? html`<a href=${'https://arxiv.org/abs/' + pid}><${MathText} text=${center.title || pid} /></a>` : pid}</h1>
            <p class="muted small">Time runs left to right: what a paper cites lies to its left, what cites it to its right.
                ${' '}Height is topic, from the papers' embeddings: related work forms bands. Hover a paper to trace its lineage; click one to spread out its citations${res && res.enrich_on && m.logged_in
                ? ', or to have the server read its references if nobody has yet (about a minute)' : ''}.</p>
        </section>
        <div class="graph-wrap">
            <div class="graph-canvas">
                ${!res ? html`<div class="graph-loading muted">Gathering citations…</div>` : null}
                <svg ref=${svgRef} class="graph-svg" onPointerMove=${move} onPointerUp=${up} onPointerCancel=${up}
                     onPointerDown=${e => down(e, null)} role="img" aria-label="citation graph">
                    <g transform=${`translate(${v.x},${v.y}) scale(${v.k})`}>
                        ${g ? g.years.map((y, i) => html`<rect key=${'b' + y} class=${'graph-band' + (i % 2 ? ' odd' : '')}
                            x=${g.yearX(y) - YEAR_GAP / 2} y="-20000" width=${YEAR_GAP} height="40000" />`) : null}
                        ${g ? [...g.edges.values()].map(([ia, ib]) => {
                            const a = g.nodes.get(ia), b = g.nodes.get(ib);
                            const mx = (a.x + b.x) / 2;
                            const onPath = focus && lit(ia) && lit(ib) && (up_.has(ia) && up_.has(ib) || down_.has(ia) && down_.has(ib));
                            const cls = 'graph-edge ' + side(ia === pid ? ib : ia) + (onPath ? ' on' : focus ? ' dim' : '');
                            return html`<path key=${ia + '>' + ib} class=${cls} pathLength="1"
                                d=${`M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`} />`;
                        }) : null}
                        ${g ? ghostList.map(n => {
                            const a = g.nodes.get(pid);
                            return a ? html`<line key=${'~' + n.id} class=${'graph-thread' + (focus && focus !== n.id && focus !== pid ? ' dim' : '')}
                                x1=${a.x} y1=${a.y} x2=${n.x} y2=${n.y} />` : null;
                        }) : null}
                        ${ghostList.map(n => html`<g key=${n.id} class=${'graph-node graph-ghost' + (focus && focus !== n.id && focus !== pid ? ' dim' : '') + (n.id === selected ? ' selected' : '')}
                                transform=${`translate(${n.x},${n.y})`} onPointerDown=${e => down(e, n)}
                                onPointerEnter=${() => setHover(n.id)} onPointerLeave=${() => setHover(x => (x === n.id ? null : x))}>
                            <circle class="graph-dot" r=${7 * grow} />
                            ${labeled.has(n.id) ? html`<text class="graph-label" y=${-7 * grow - 6 / v.k}
                                style=${{ fontSize: 12 / v.k + 'px' }}>${short(n.title || n.id)}</text>` : null}
                        </g>`)}
                        ${nodes.map(n => {
                            const r = radius(n.id, cited, n.id === pid) * grow;
                            const cls = 'graph-node ' + side(n.id) + ' ' + n.state + (n.saved.length ? ' saved' : '') + (n.here ? '' : ' away')
                                + (n.id === selected ? ' selected' : '') + (lit(n.id) ? '' : ' dim') + (now - n.born < 900 ? ' fresh' : '');
                            return html`<g key=${n.id} class=${cls} transform=${`translate(${n.x},${n.y})`}
                                    onPointerDown=${e => down(e, n)} onPointerEnter=${() => setHover(n.id)}
                                    onPointerLeave=${() => setHover(x => (x === n.id ? null : x))}>
                                ${n.saved.length ? html`<circle class="graph-halo" r=${r + 5} />` : null}
                                <circle class="graph-dot" r=${r} />
                                ${n.state === 'queued' ? html`<circle class="graph-scan" r=${r + 4} pathLength="1" />` : null}
                                ${labeled.has(n.id) ? html`<text class="graph-label" y=${-r - 6 / v.k} style=${{ fontSize: 12 / v.k + 'px' }}>
                                    ${short(n.title || n.id, n.id === pid ? 60 : 38)}</text>` : null}
                            </g>`;
                        })}
                    </g>
                    ${g ? html`<g class="graph-years">
                        <rect x="0" y="0" width="100%" height="24" class="graph-ruler" />
                        ${rulerYears(g, v).map(([y, x]) => html`<text key=${y} x=${x} y="16">${y}</text>`)}
                    </g>` : null}
                </svg>
                ${hov && !drag.current ? html`<div class="graph-tip" style=${{
                    left: Math.min(hov.x * v.k + v.x + 14, (svgRef.current ? svgRef.current.clientWidth : 600) - 260) + 'px',
                    top: Math.min(hov.y * v.k + v.y + 14, h - 90) + 'px' }}>
                    <b><${MathText} text=${hov.title || hov.id} /></b>
                    <div class="muted">${hov.year}${hov.authors ? ' · ' + hov.authors : ''} · ${STATE_TEXT[hov.state]}</div>
                    <div class="muted">${hov.ghost ? Math.round(100 * hov.similarity) + '% alike in content, but no citation links it here. Click to add it.'
                        : 'cited by ' + (cited.get(hov.id) || 0) + ' here' + ((out.get(hov.id) || []).length ? ' · cites ' + out.get(hov.id).length + ' here' : '')}</div>
                </div>` : null}
                <div class="graph-zoom">
                    <button type="button" class="btn btn-small" title="zoom in" aria-label="zoom in"
                        onClick=${() => zoomAt(svgRef.current.clientWidth / 2, svgRef.current.clientHeight / 2, 1.3)}>+</button>
                    <button type="button" class="btn btn-small" title="zoom out" aria-label="zoom out"
                        onClick=${() => zoomAt(svgRef.current.clientWidth / 2, svgRef.current.clientHeight / 2, 1 / 1.3)}>−</button>
                    <button type="button" class="btn btn-small" title="show the whole graph" onClick=${() => { fitted.current = true; fit(); }}>Fit</button>
                </div>
                ${g && res ? html`<div class="graph-count">
                    <span><i class="lg past"></i>built on</span><span><i class="lg center"></i>this paper</span><span><i class="lg future"></i>built on it</span>
                    <span><i class="lg hollow"></i>not read yet</span><span><i class="lg saved"></i>yours</span>
                    ${ghostList.length ? html`<span><i class="lg ghost"></i>alike, not linked</span>` : null}
                    <span class="muted">${nodes.length} papers · ${g.edges.size} citations${reading ? ' · ' + reading + ' being read' : ''}</span>
                </div>` : null}
            </div>
            ${res && g ? html`<aside class="graph-side">
                ${sel ? html`<${Selected} n=${sel} res=${res} m=${m} queue=${queue} cited=${cited} out=${out} busy=${busy} error=${actionError}
                    onExpand=${expand} onAnalyze=${analyze} isCenter=${sel.id === pid} />` : null}
                ${ghostList.length ? html`<div class="card graph-top graph-ghosts">
                    <h3>Alike, but not linked</h3>
                    <p class="muted small">Closest to this paper in content (by their embeddings), yet no citation connects them:
                        related work it missed, or that missed it.</p>
                    <ol>${ghostList.map(n => html`<li key=${n.id}>
                        <button type="button" class="linkbtn" onClick=${() => clickNode(n)}
                            onPointerEnter=${() => setHover(n.id)} onPointerLeave=${() => setHover(null)}>${short(n.title || n.id, 60)}</button>
                        <div class="muted small">${n.year}${n.authors ? ' · ' + n.authors : ''} · ${Math.round(100 * n.similarity)}% alike</div></li>`)}</ol>
                </div>` : null}
                ${top.length ? html`<div class="card graph-top">
                    <h3>Most cited here</h3>
                    <p class="muted small">The papers this neighbourhood keeps citing: often the area's foundations.</p>
                    <ol>${top.slice(0, 8).map(n => html`<li key=${n.id}>
                        <button type="button" class="linkbtn" onClick=${() => clickNode(n)}
                            onPointerEnter=${() => setHover(n.id)} onPointerLeave=${() => setHover(null)}>${short(n.title || n.id, 60)}</button>
                        <span class="muted"> · ${n.year} · ×${cited.get(n.id)}</span></li>`)}</ol>
                </div>` : null}
            </aside>` : null}
        </div>
    </div>`;
};

// the selected paper: what it is, and what can be done with it
const Selected = ({ n, res, m, queue, cited, out, busy, error, onExpand, onAnalyze, isCenter }) => {
    const pos = queue.indexOf(n.id);
    return html`
    <div class=${'card graph-panel ' + n.state}>
        <div class="graph-panel-head muted small">
            <span>${n.year}${n.authors ? ' · ' + n.authors : ''}</span>
            ${n.saved.length ? html`<span class="graph-saved" title="in your tags">★ ${n.saved.join(', ')}</span>` : null}
        </div>
        <h2 class="graph-title"><a href=${'https://arxiv.org/abs/' + n.id}><${MathText} text=${n.title || n.id} /></a></h2>
        <p class="muted small">
            ${STATE_TEXT[n.state]}${n.state === 'analyzed' && !n.readable ? ', but no reference list was found in it' : ''}
            ${n.state === 'queued' && pos >= 0 ? ' (number ' + (pos + 1) + ' in line, about ' + Math.max(1, Math.round(((pos + 1) * SECONDS_PER_PAPER) / 60)) + ' min)' : ''}.
            ${' '}Here it is cited by ${cited.get(n.id) || 0}${(out.get(n.id) || []).length ? ' and cites ' + out.get(n.id).length : ''}.</p>
        <${Err} msg=${error} />
        <div class="graph-actions">
            ${n.state === 'analyzed' && !n.expanded ? html`<button type="button" class="btn btn-small btn-primary" disabled=${busy}
                onClick=${() => onExpand(n.id)} title="add the papers it cites and the ones citing it">Spread out its citations</button>` : null}
            ${n.state === 'new' && res.enrich_on && m.logged_in ? html`<button type="button" class="btn btn-small btn-primary" disabled=${busy}
                onClick=${() => onAnalyze(n.id)} title="the server reads its reference list from arXiv, about a minute">Read its references</button>` : null}
            ${n.state === 'new' && !m.logged_in ? html`<span class="muted small"><a href="/profile">Log in</a> to have its references read.</span>` : null}
            ${!isCenter ? html`<a class="btn btn-small" href=${'/graph?pid=' + n.id} title="a new graph around this paper">Start from here</a>` : null}
            ${n.here ? html`<a class="btn btn-small" href=${'/?rank=pid&pid=' + n.id} title="papers like it, by their text">Similar</a>` : null}
            <a class="btn btn-small" href=${'https://arxiv.org/pdf/' + n.id}>pdf</a>
        </div>
    </div>`;
};
