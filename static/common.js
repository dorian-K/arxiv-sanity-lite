'use strict';
// shared helpers for the React components: htm instead of JSX, and lazy LaTeX rendering

const html = htm.bind(React.createElement);
const { useState, useEffect, useRef } = React;

function mount(component, elementId) {
    const elt = document.getElementById(elementId);
    if (elt) {
        ReactDOM.createRoot(elt).render(component);
    }
}

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

// text that may contain LaTeX. React never manages the children of the span, because
// KaTeX replaces them, so the text is set by hand and typeset lazily
const MathText = ({ text, tag = 'span', className }) => {
    const ref = useRef(null);
    useEffect(() => {
        const elt = ref.current;
        elt.textContent = text;
        if (!HAS_MATH.test(text)) return;
        if (mathObserver) {
            mathObserver.observe(elt);
            return () => mathObserver.unobserve(elt);
        }
        typeset(elt);
    }, [text]);
    return React.createElement(tag, { ref, className });
};

// -----------------------------------------------------------------------------
// a loading indicator while the next page is on its way: the old page stays usable, and
// after a moment a bar and a line about what is happening show up. a form can say what
// it is waiting for with data-busy="…"

let busyTimer = null;

function showBusy(msg) {
    clearTimeout(busyTimer);
    busyTimer = setTimeout(() => {
        let elt = document.getElementById('busy');
        if (!elt) {
            elt = document.createElement('div');
            elt.id = 'busy';
            elt.setAttribute('role', 'status');
            elt.innerHTML = '<div class="busy-bar"></div><div class="busy-msg"></div>';
            document.body.appendChild(elt);
        }
        elt.querySelector('.busy-msg').textContent = msg;
        elt.hidden = false;
    }, 300); // fast pages come back before anything flashes
}

function hideBusy() {
    clearTimeout(busyTimer);
    const elt = document.getElementById('busy');
    if (elt) elt.hidden = true;
}

function busyMessage(url) {
    const q = url.searchParams;
    if (url.pathname === '/' && q.get('q')) return 'Searching…';
    if (url.pathname === '/digest') return 'Loading your digest…';
    return 'Loading…';
}

// back/forward restores the page from the cache as it was left, indicator included
window.addEventListener('pageshow', hideBusy);

document.addEventListener('click', e => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest('a[href]');
    if (!a || a.target || a.hasAttribute('download')) return;
    const url = new URL(a.href, window.location.href);
    if (url.origin !== window.location.origin || url.pathname.startsWith('/static/')) return;
    if (url.hash && url.pathname === window.location.pathname && url.search === window.location.search) return;
    showBusy(busyMessage(url));
});

document.addEventListener('submit', e => {
    const form = e.target;
    // React forms (tag picker) handle their submit themselves
    if (e.defaultPrevented || form.target) return;
    const url = new URL(form.action, window.location.href);
    if (form.method.toLowerCase() === 'get') {
        url.search = new URLSearchParams(new FormData(form)).toString();
    }
    showBusy(form.dataset.busy || busyMessage(url));
});
