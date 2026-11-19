// where a search matched, lightly marked on the cards (titles, authors, abstracts, notes) with the
// CSS Custom Highlight API: nothing in the page changes, so it can't break the math typesetting
// (browsers without it just show no marks). the words the search used match at the start of a
// word, as in the search ('diffus' marks diffusion); stopwords only inside the query's own phrases
// ("learning to rank", "the transformer"), with the query's words around them, never on their own
const WORD = '[\\p{L}\\p{N}]';
const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const one = (w, stop) => esc(w) + (stop || w.length < 3 ? `(?!${WORD})` : `${WORD}*`);

const matchers = new Map();

// terms: [[word, isStop], ...] from the server (serve.search_highlight); the same RegExp for the same terms
export function matcher(terms) {
    if (!terms || !terms.length || !terms.some(t => !t[1])) return null;
    const key = JSON.stringify(terms);
    if (!matchers.has(key)) {
        const alts = [];
        for (let i = 0; i < terms.length; i++) {
            for (let j = i + 1; j < terms.length; j++) {
                const seq = terms.slice(i, j + 1);
                // a stopword needs its neighbours from the query on both sides (or the query's start or end)
                const ends = (i === 0 || !seq[0][1]) && (j === terms.length - 1 || !seq[seq.length - 1][1]);
                if (ends && seq.some(t => t[1]) && seq.some(t => !t[1])) alts.push(seq.map(t => one(t[0], t[1])).join('[\\s–-]+'));
            }
        }
        for (const [w, stop] of terms) if (!stop) alts.push(one(w, false));
        alts.sort((a, b) => b.length - a.length); // the longest exact phrase wins
        try {
            matchers.set(key, new RegExp(`(?<!${WORD})(?:${[...new Set(alts)].join('|')})`, 'giu'));
        } catch (e) {
            matchers.set(key, null);
        }
    }
    return matchers.get(key);
}

const MATH = /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|(?<!\\)\$(?:[^$\\]|\\[\s\S])+?\$/g;
const marked = new Map(); // element -> RegExp
let timer = null;
const supported = () => typeof CSS !== 'undefined' && CSS.highlights && typeof Highlight !== 'undefined';

// marks the matches of re in elt (null: no longer); call again after its content changed
export function mark(elt, re) {
    if (!elt || !supported()) return;
    if (re) marked.set(elt, re);
    else marked.delete(elt);
    repaint();
}

export function repaint() {
    if (!supported()) return;
    clearTimeout(timer);
    timer = setTimeout(paint, 30);
}

function paint() {
    const hl = new Highlight();
    for (const [elt, re] of marked) {
        if (!elt.isConnected) { marked.delete(elt); continue; }
        const walker = document.createTreeWalker(elt, NodeFilter.SHOW_TEXT, {
            acceptNode: n => (n.parentElement && n.parentElement.closest('.katex, code, pre, button') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
        });
        for (let n = walker.nextNode(); n; n = walker.nextNode()) {
            const text = n.data;
            const math = []; // raw LaTeX not typeset yet: no marks inside it
            if (text.includes('$') || text.includes('\\')) for (const m of text.matchAll(MATH)) math.push([m.index, m.index + m[0].length]);
            re.lastIndex = 0;
            for (let m = re.exec(text); m; m = re.exec(text)) {
                if (!m[0]) { re.lastIndex++; continue; }
                const a = m.index, b = a + m[0].length;
                if (math.some(([s, e]) => a < e && b > s)) continue;
                const r = new Range();
                r.setStart(n, a);
                r.setEnd(n, b);
                hl.add(r);
            }
        }
    }
    CSS.highlights.set('search', hl);
}
