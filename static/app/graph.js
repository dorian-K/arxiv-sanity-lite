// the citation graph (/graph?pid=): a paper, the papers it cites and the ones citing it, drawn
// as a force layout in SVG. clicking a paper whose references are known adds its neighbours;
// one that hasn't been read yet is asked for (/graph/analyze) and added once the server has read it
import { html, useState, useEffect, useRef, me, api, location, Err } from 'app/lib';
import { MathText } from 'app/math';

const LINK = 70; // spring length between a paper and what it cites
const REPULSE = 900;
const POLL = 6000; // ms between looks at papers waiting to be read
const SECONDS_PER_PAPER = 60; // enrich_daemon.py: up to three requests to arxiv.org, 20 s apart

// the graph's state outside Preact: positions change every frame
class Graph {
    constructor() {
        this.nodes = new Map(); // id -> {id, title, ..., x, y, vx, vy, expanded}
        this.edges = new Map(); // 'a>b' -> [a, b]: a cites b
        this.alpha = 1;
    }

    // a response of /ui/graph: new papers start next to the one they came from
    merge(res, from) {
        const origin = this.nodes.get(from) || { x: 0, y: 0 };
        for (const n of res.nodes) {
            const old = this.nodes.get(n.id);
            if (old) Object.assign(old, n, { expanded: old.expanded });
            else {
                const a = Math.random() * 2 * Math.PI, r = 30 + Math.random() * 40;
                this.nodes.set(n.id, { ...n, x: origin.x + r * Math.cos(a), y: origin.y + r * Math.sin(a), vx: 0, vy: 0 });
            }
        }
        for (const [a, b] of res.edges) this.edges.set(a + '>' + b, [a, b]);
        this.nodes.get(res.pid).expanded = true;
        this.alpha = Math.max(this.alpha, 0.8);
    }

    degree() {
        const d = new Map();
        for (const [a, b] of this.edges.values()) {
            d.set(a, (d.get(a) || 0) + 1);
            d.set(b, (d.get(b) || 0) + 1);
        }
        return d;
    }

    // one step of the layout: papers push each other away, citations pull them together
    tick(dragging) {
        if (this.alpha < 0.01) return false;
        const ns = [...this.nodes.values()];
        const k = this.alpha;
        for (let i = 0; i < ns.length; i++) {
            const a = ns[i];
            for (let j = i + 1; j < ns.length; j++) {
                const b = ns[j];
                let dx = a.x - b.x, dy = a.y - b.y;
                let d2 = dx * dx + dy * dy;
                if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
                if (d2 > 250000) continue; // far apart: no effect worth the time
                const f = (REPULSE * k) / d2;
                a.vx += dx * f; a.vy += dy * f;
                b.vx -= dx * f; b.vy -= dy * f;
            }
        }
        for (const [ia, ib] of this.edges.values()) {
            const a = this.nodes.get(ia), b = this.nodes.get(ib);
            const dx = b.x - a.x, dy = b.y - a.y;
            const d = Math.sqrt(dx * dx + dy * dy) || 1;
            const f = ((d - LINK) / d) * 0.06 * k;
            a.vx += dx * f; a.vy += dy * f;
            b.vx -= dx * f; b.vy -= dy * f;
        }
        for (const n of ns) {
            n.vx -= n.x * 0.004 * k; n.vy -= n.y * 0.004 * k; // towards the middle
            if (n === dragging) { n.vx = 0; n.vy = 0; continue; }
            n.vx *= 0.55; n.vy *= 0.55;
            n.x += n.vx; n.y += n.vy;
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

// in graph units; drawn a bit bigger when zoomed out, so papers never shrink to dots
const radius = (n, deg, center) => (center ? 11 : 5 + 2.5 * Math.sqrt(deg.get(n.id) || 0));
const short = (t, n = 34) => (t.length > n ? t.slice(0, n - 1) + '…' : t);

const STATE_TEXT = {
    analyzed: 'References read',
    queued: 'Waiting to be read',
    failed: "Its references couldn't be read",
    new: 'References not read yet',
};

// the selected paper: what it is, and what can be done with it
const Panel = ({ n, g, res, onExpand, onAnalyze, busy, error }) => {
    const m = me.use();
    if (!n) return null;
    const queue = res.queue || [];
    const pos = queue.indexOf(n.id);
    const cites = [...g.edges.values()].filter(([a]) => a === n.id).length;
    const cited = [...g.edges.values()].filter(([, b]) => b === n.id).length;
    return html`
    <div class="graph-panel card">
        <div class="graph-panel-head">
            <span class="muted">${n.year}${n.authors ? ' · ' + n.authors : ''}</span>
            ${n.saved.length ? html`<span class="graph-saved" title="in your tags">★ ${n.saved.join(', ')}</span>` : null}
        </div>
        <h2 class="graph-title"><a href=${'https://arxiv.org/abs/' + n.id}><${MathText} text=${n.title || n.id} /></a></h2>
        <p class="muted small">${STATE_TEXT[n.state]}${n.state === 'analyzed' && !n.readable ? ', but no reference list was found in it' : ''}.
            ${' '}On the graph: cites ${cites}, cited by ${cited}${n.state === 'analyzed' ? '' : ' (that we know of)'}.
            ${n.state === 'queued' && pos >= 0 ? ' Number ' + (pos + 1) + ' in line, about ' + Math.max(1, Math.round(((pos + 1) * SECONDS_PER_PAPER) / 60)) + ' min.' : ''}</p>
        <${Err} msg=${error} />
        <div class="graph-actions">
            ${n.state === 'analyzed' && !n.expanded ? html`<button type="button" class="btn btn-small btn-primary" disabled=${busy}
                onClick=${() => onExpand(n.id)} title="add the papers it cites and the ones citing it">Show its citations</button>` : null}
            ${n.state === 'new' && res.enrich_on && m.logged_in ? html`<button type="button" class="btn btn-small btn-primary" disabled=${busy}
                onClick=${() => onAnalyze(n.id)} title="the server reads its reference list from arXiv, about a minute">Read its references</button>` : null}
            ${n.state === 'new' && !m.logged_in ? html`<span class="muted small"><a href="/profile">Log in</a> to have its references read.</span>` : null}
            ${n.id !== res.pid ? html`<a class="btn btn-small" href=${'/graph?pid=' + n.id} title="start a new graph from this paper">Start from here</a>` : null}
            ${n.here ? html`<a class="btn btn-small" href=${'/?rank=pid&pid=' + n.id} title="papers like it, by their text">Similar papers</a>` : null}
        </div>
    </div>`;
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

    const redraw = () => setFrame(f => f + 1);

    // the layout runs while it moves, and stops when it has settled
    const running = useRef(false);
    const run = () => {
        if (running.current) return;
        running.current = true;
        const step = () => {
            const g = gRef.current;
            if (!g) { running.current = false; return; }
            const moved = g.tick(drag.current && drag.current.node);
            redraw();
            if (moved || drag.current) requestAnimationFrame(step);
            else running.current = false;
        };
        requestAnimationFrame(step);
    };

    const fit = () => {
        const svg = svgRef.current, g = gRef.current;
        if (!svg || !g) return;
        const [x0, y0, x1, y1] = g.bounds();
        const w = svg.clientWidth, h = svg.clientHeight;
        const k = Math.min(2, 0.9 * Math.min(w / (x1 - x0 + 80), h / (y1 - y0 + 80)));
        view.current = { k, x: w / 2 - ((x0 + x1) / 2) * k, y: h / 2 - ((y0 + y1) / 2) * k };
        redraw();
    };

    const fetchAround = id => api('/ui/graph?pid=' + encodeURIComponent(id));

    // a new center: a new graph
    useEffect(() => {
        gRef.current = new Graph();
        setRes(null); setError(''); setSelected(pid); setActionError('');
        fetchAround(pid).then(r => {
            const g = gRef.current;
            g.merge(r, pid);
            setRes(r);
            const c = g.nodes.get(pid);
            c.x = 0; c.y = 0;
            // the paper this graph is about: read it right away if it hasn't been yet
            if (c.state === 'new' && r.enrich_on && me.get().logged_in) analyze(pid);
            run();
            setTimeout(fit, 700);
            setTimeout(fit, 2200);
        }).catch(e => setError(e.message));
        return () => { gRef.current = null; };
    }, [pid]);

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

    // papers waiting to be read: looked at every few seconds, and spread out once they are
    useEffect(() => {
        const timer = setInterval(() => {
            const g = gRef.current;
            if (!g) return;
            const waiting = [...g.nodes.values()].filter(n => n.state === 'queued').slice(0, 5);
            for (const n of waiting) {
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
        if (n.state === 'analyzed' && !n.expanded) expand(n.id);
        else if (n.state === 'new' && res && res.enrich_on && m.logged_in) analyze(n.id);
    };

    // pointer handling: drag a paper, drag the background to pan, wheel or pinch to zoom
    const toGraph = e => {
        const r = svgRef.current.getBoundingClientRect(), v = view.current;
        return [(e.clientX - r.left - v.x) / v.k, (e.clientY - r.top - v.y) / v.k];
    };
    const zoomAt = (cx, cy, factor) => {
        const v = view.current;
        const k = Math.min(6, Math.max(0.15, v.k * factor));
        view.current = { k, x: cx - ((cx - v.x) * k) / v.k, y: cy - ((cy - v.y) * k) / v.k };
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
        if (!d) { return; }
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
    };

    if (error) return html`<div class="page"><section class="card"><p class="form-error">${error}</p></section></div>`;
    const g = gRef.current;
    const center = g && g.nodes.get(pid);
    const deg = g ? g.degree() : new Map();
    const v = view.current;
    const grow = Math.max(1, 1 / Math.sqrt(v.k));
    const sel = g && g.nodes.get(selected);
    // labels for the paper itself, the selected and hovered ones, and the most connected
    const labeled = new Set([pid, selected, hover]);
    // (papers several others cite, often the field's key references)
    if (g) {
        [...g.nodes.values()].filter(n => (deg.get(n.id) || 0) >= 3 && n.id !== pid)
            .sort((a, b) => deg.get(b.id) - deg.get(a.id)).slice(0, 8).forEach(n => labeled.add(n.id));
    }
    const near = id => selected && (id === selected);
    return html`
    <div class="graph-page">
        <section class="graph-head">
            <h1>Citation graph${center ? html`: <a href=${'https://arxiv.org/abs/' + pid}><${MathText} text=${short(center.title || pid, 90)} /></a>` : null}</h1>
            <p class="muted small">Arrows point from a paper to the papers it cites. Filled circles are papers whose
                reference list this site has read: click one to add what it cites and what cites it. For a hollow one,
                ${' '}${res && !res.enrich_on ? 'its references would have to be read first, which is turned off on this server.'
                : m.logged_in ? 'a click has the server read its references from arXiv (about a minute; one request every 20 s, to be polite).'
                : html`<a href="/profile">log in</a> to have its references read.`}
                ${' '}Drag to move, scroll or pinch to zoom.</p>
            <div class="graph-legend muted small">
                <span><i class="lg lg-center"></i>this paper</span>
                <span><i class="lg lg-analyzed"></i>references read</span>
                <span><i class="lg lg-new"></i>not read yet</span>
                <span><i class="lg lg-saved"></i>in your tags</span>
                <span><i class="lg lg-away"></i>not on this site</span>
            </div>
        </section>
        <div class="graph-wrap">
            <div class="graph-canvas">
                ${!res ? html`<div class="graph-loading muted">Loading the graph…</div>` : null}
                <svg ref=${svgRef} class="graph-svg" onPointerMove=${move} onPointerUp=${up} onPointerCancel=${up}
                     onPointerDown=${e => down(e, null)} role="img" aria-label="citation graph">
                    <defs>
                        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" class="graph-arrow" />
                        </marker>
                        <marker id="arrow-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" class="graph-arrow on" />
                        </marker>
                    </defs>
                    <g transform=${`translate(${v.x},${v.y}) scale(${v.k})`}>
                        ${g ? [...g.edges.values()].map(([ia, ib]) => {
                            const a = g.nodes.get(ia), b = g.nodes.get(ib);
                            const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
                            const rb = radius(b, deg, ib === pid) * grow + 2;
                            const on = near(ia) || near(ib);
                            return html`<line key=${ia + '>' + ib} x1=${a.x} y1=${a.y} x2=${b.x - (dx / d) * rb} y2=${b.y - (dy / d) * rb}
                                class=${'graph-edge' + (on ? ' on' : '')} marker-end=${on ? 'url(#arrow-on)' : 'url(#arrow)'} />`;
                        }) : null}
                        ${g ? [...g.nodes.values()].map(n => {
                            const c = n.id === pid;
                            const cls = 'graph-node ' + n.state + (c ? ' center' : '') + (n.saved.length ? ' saved' : '')
                                + (n.here ? '' : ' away') + (n.id === selected ? ' selected' : '');
                            return html`<g key=${n.id} class=${cls} transform=${`translate(${n.x},${n.y})`}
                                    onPointerDown=${e => down(e, n)} onPointerEnter=${() => setHover(n.id)} onPointerLeave=${() => setHover(h => (h === n.id ? null : h))}>
                                <title>${(n.title || n.id) + (n.year ? ' (' + n.year + ')' : '')}</title>
                                <circle r=${radius(n, deg, c) * grow} />
                                ${labeled.has(n.id) ? html`<text x=${radius(n, deg, c) * grow + 4 / v.k} y=${4 / v.k} style=${{ fontSize: 12 / v.k + 'px' }}>
                                    ${short(n.title || n.id)}${n.year ? ' ' + n.year : ''}</text>` : null}
                            </g>`;
                        }) : null}
                    </g>
                </svg>
                <div class="graph-zoom">
                    <button type="button" class="btn btn-small" title="zoom in" aria-label="zoom in"
                        onClick=${() => zoomAt(svgRef.current.clientWidth / 2, svgRef.current.clientHeight / 2, 1.3)}>+</button>
                    <button type="button" class="btn btn-small" title="zoom out" aria-label="zoom out"
                        onClick=${() => zoomAt(svgRef.current.clientWidth / 2, svgRef.current.clientHeight / 2, 1 / 1.3)}>−</button>
                    <button type="button" class="btn btn-small" title="show the whole graph" onClick=${fit}>Fit</button>
                </div>
                ${g && res ? html`<div class="graph-count muted small">${g.nodes.size} papers, ${g.edges.size} citations</div>` : null}
            </div>
            ${res && g ? html`<${Panel} n=${sel} g=${g} res=${res} busy=${busy} error=${actionError}
                onExpand=${expand} onAnalyze=${analyze} />` : null}
        </div>
    </div>`;
};
