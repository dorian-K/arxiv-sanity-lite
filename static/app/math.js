// LaTeX in titles, abstracts and summaries: text-mode LaTeX turned into html, and math typeset by
// KaTeX, which is only downloaded once the first piece of math scrolls into view
import { h, useRef, useEffect } from 'app/lib';

// -----------------------------------------------------------------------------
// lazy LaTeX: KaTeX is only downloaded once the first piece of text containing math
// scrolls into view, and each piece of text is typeset when it becomes visible.

const KATEX = '/static/vendor/katex-0.16.11/';  // served from this server, see update_vendor.py
const MATH_DELIMITERS = [
    { left: '$$', right: '$$', display: true },
    { left: '\\[', right: '\\]', display: true },
    { left: '$', right: '$', display: false },
    { left: '\\(', right: '\\)', display: false },
];
const HAS_MATH = /\$[^$]+\$|\\\(|\\\[/;

let katexPromise = null;
function loadKatex() {
    if (!katexPromise) {
        const css = document.createElement('link');
        css.rel = 'stylesheet';
        css.href = KATEX + 'katex.min.css';
        document.head.appendChild(css);
        const script = src => new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = src;
            s.onload = resolve;
            s.onerror = reject;
            document.head.appendChild(s);
        });
        katexPromise = script(KATEX + 'katex.min.js').then(() => script(KATEX + 'contrib/auto-render.min.js'));
    }
    return katexPromise;
}

const mathObserver = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        mathObserver.unobserve(entry.target);
        typeset(entry.target);
    }
}, { rootMargin: '400px 0px' }) : null; // start a bit before the text is on screen

function typeset(elt) {
    loadKatex().then(() => {
        renderMathInElement(elt, { delimiters: MATH_DELIMITERS, throwOnError: false });
    }).catch(() => {}); // offline or blocked CDN: leave the raw LaTeX
}

// LaTeX that isn't math (\emph{…}, {\bf …}, \%, ``…'', \cite{…}, accents) is common in arXiv
// abstracts, and KaTeX only typesets math. textLatex turns it into html, leaving the math
// pieces as they are for KaTeX; everything is escaped first, so only the tags made here exist
const HAS_TEXT_LATEX = /\\[a-zA-Z]+|\\[%&_#$'"`^~]|``|\{\\(em|it|bf|sc|tt|sl)\b/;
const MATH_PIECES = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|(?<!\\)\$(?:[^$\\]|\\[\s\S])+?\$)/;
const TEXT_STYLE = { emph: 'em', textit: 'em', textsl: 'em', textbf: 'b', textsc: 'span class="smallcaps"', texttt: 'code', underline: 'u' };
const OLD_STYLE = { em: 'em', it: 'em', sl: 'em', bf: 'b', sc: 'span class="smallcaps"', tt: 'code' };
const SYMBOLS = {
    alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ',
    kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', phi: 'φ',
    varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Pi: 'Π',
    Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω', deg: '°', times: '×', approx: '≈', sim: '~', le: '≤', leq: '≤',
    ge: '≥', geq: '≥', neq: '≠', pm: '±', infty: '∞', cdot: '·', ldots: '…', dots: '…', to: '→', rightarrow: '→',
    in: '∈', sqrt: '√', log: 'log', textasciitilde: '~', LaTeX: 'LaTeX', TeX: 'TeX', ie: 'i.e.', eg: 'e.g.',
};
const ACCENTS = { "'": '́', '`': '̀', '"': '̈', '^': '̂', '~': '̃', c: '̧', v: '̌', '=': '̄', u: '̆', H: '̋' };
// arXiv's metadata sometimes has unicode letters inside math ($η_t$), which KaTeX refuses
const MATH_UNICODE = {};
for (const [cmd, ch] of Object.entries(SYMBOLS)) {
    if (ch.length === 1 && !(ch in MATH_UNICODE) && !['sim', 'cdot', 'ldots', 'dots', 'textasciitilde', 'deg'].includes(cmd)) MATH_UNICODE[ch] = '\\' + cmd + ' ';
}
const mathUnicode = m => m.replace(/[^\x00-\x7f]/g, ch => MATH_UNICODE[ch] ?? ch);
const escapeHtml = t => t.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
export function textLatex(text) {
    return text.split(MATH_PIECES).map((piece, i) => {
        if (i % 2) return escapeHtml(mathUnicode(piece)); // math, typeset by KaTeX afterwards
        let t = piece.replace(/\\([`'"^~=]|[cvuH](?=\{))\{?([a-zA-Z])\}?/g, (m, a, ch) => (ACCENTS[a] ? (ch + ACCENTS[a]).normalize('NFC') : m));
        t = escapeHtml(t);
        t = t.replace(/\s*\\(cite[pt]?|citeauthor|ref|eqref|label)\*?(\[[^\]]*\])?\{[^{}]*\}/g, '');
        t = t.replace(/\s*\\footnote\{([^{}]*)\}/g, ' ($1)');
        t = t.replace(/\\(?:url|href)\{([^{}]*)\}(?:\{([^{}]*)\})?/g, (m, u, label) => label || u);
        for (let round = 0; round < 3; round++) { // nested ones, from the inside out
            t = t.replace(/\\([a-zA-Z]+)\{([^{}]*)\}/g, (m, cmd, body) => {
                const tag = TEXT_STYLE[cmd];
                return tag ? '<' + tag + '>' + body + '</' + tag.split(' ')[0] + '>' : cmd === 'mbox' || cmd === 'text' ? body : m;
            });
            t = t.replace(/\{\\(em|it|bf|sc|tt|sl)\b\s*([^{}]*)\}/g, (m, cmd, body) => {
                const tag = OLD_STYLE[cmd];
                return '<' + tag + '>' + body + '</' + tag.split(' ')[0] + '>';
            });
        }
        t = t.replace(/``/g, '“').replace(/''/g, '”');
        t = t.replace(/\\([a-zA-Z]+)\b;?/g, (m, cmd) => SYMBOLS[cmd] ?? m);
        t = t.replace(/\\([%_#&])/g, '$1').replace(/\\\$/g, '<span>$</span>');
        t = t.replace(/\\[ ,;!]/g, ' ');
        return t.replace(/(?<!\\[a-zA-Z]+)\{([^{}$\\]*)\}/g, '$1'); // grouping braces left over
    }).join('');
}

// text that may contain LaTeX. Preact never manages the children of the span, because
// KaTeX replaces them, so the text is set by hand and typeset lazily
export const MathText = ({ text, tag = 'span', className }) => {
    const ref = useRef(null);
    useEffect(() => {
        const elt = ref.current;
        if (HAS_TEXT_LATEX.test(text) || HAS_MATH.test(text)) elt.innerHTML = textLatex(text);
        else elt.textContent = text;
        if (!HAS_MATH.test(text)) return;
        if (mathObserver) {
            mathObserver.observe(elt);
            return () => mathObserver.unobserve(elt);
        }
        typeset(elt);
    }, [text]);
    return h(tag, { ref, className });
};
