// the app's plumbing: htm bound to Preact, the server's /ui/ data, who is logged in, the
// router (links switch pages without loading a new document), and the busy indicator
import { h, createContext } from 'preact';
import { useState, useEffect, useRef, useContext } from 'preact/hooks';
import htm from 'htm';

export const html = htm.bind(h);
export { h, useState, useEffect, useRef, useContext, createContext };

// -----------------------------------------------------------------------------
// small stores: a value, and the components that show it re-render when it changes

export function store(value) {
    const listeners = new Set();
    return {
        get: () => value,
        set(v) { value = v; listeners.forEach(f => f()); },
        use() {
            const [, bump] = useState(0);
            useEffect(() => {
                const f = () => bump(n => n + 1);
                listeners.add(f);
                return () => listeners.delete(f);
            }, []);
            return value;
        },
    };
}

// -----------------------------------------------------------------------------
// talking to the server. the app's requests say Accept: application/json, so errors come back
// as {error}; the vote endpoints (/add, ...) answer plain text, "ok..." or the reason

export class ApiError extends Error {}

export async function api(path, form) {
    const opts = { headers: { Accept: 'application/json' } };
    if (form !== undefined) {
        opts.method = 'POST';
        opts.body = new URLSearchParams(form);
        changed();
    }
    let r;
    try {
        r = await fetch(path, opts);
    } catch (e) {
        throw new ApiError(form !== undefined ? 'Could not reach the server, so this was not saved.' : 'Could not reach the server.');
    }
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch (e) { data = null; }
    if (!r.ok || (data && data.error)) throw new ApiError((data && data.error) || text || r.statusText);
    return data ?? text;
}

// calls one of the vote endpoints; resolves on "ok...", otherwise shows the server's message.
// one request at a time, in click order: votes show at once (before the server answers), so two
// quick clicks must not reach the server the other way round
let tagQueue = Promise.resolve();
export function tagRequest(path) {
    const send = () => api(path, {}).then(t => {
        if (typeof t !== 'string' || !t.startsWith('ok')) throw new ApiError(String(t));
        return t;
    });
    const req = tagQueue.then(send, send).catch(e => { alert(e.message); throw e; });
    tagQueue = req.catch(() => {});
    return req;
}

// pages fetched before, by address, for an instant back button and prefetching on hover. any
// change (a vote, a setting) clears them, since a list depends on every vote
const cache = new Map();
const CACHE_MAX = 30;
export function changed() { cache.clear(); }

export function load(path) {
    let p = cache.get(path);
    if (!p) {
        p = api(path);
        cache.set(path, p);
        p.catch(() => cache.delete(path));
        while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
    }
    return p;
}

// a page's data from /ui/...: {data, error, loading, reload}. while the next address loads, the
// page keeps showing the data it has (with the busy bar), so nothing flickers
export function useData(path, busyMsg) {
    const [state, setState] = useState({ data: null, error: null, path: null });
    const [round, setRound] = useState(0);
    useEffect(() => {
        let live = true;
        if (path === null) return undefined;
        busy(busyMsg || 'Loading…');
        load(path).then(
            data => { if (live) setState({ data, error: null, path }); },
            e => { if (live) setState(s => ({ data: s.data, error: e.message, path })); },
        ).finally(() => { idle(); if (live) pageReady(); });
        return () => { live = false; };
    }, [path, round]);
    return {
        data: state.data,
        error: state.path === path ? state.error : null,
        loading: state.path !== path,
        dataPath: state.path, // what data shows: the previous page's while the next one loads
        reload: () => { cache.delete(path); setRound(n => n + 1); },
    };
}

// -----------------------------------------------------------------------------
// who is looking: their name, whether read-only, the site's switches (/ui/me)

export const me = store(null);
export const refreshMe = () => api('/ui/me').then(m => { changed(); me.set(m); return m; });
export const canEdit = (m, page) => Boolean(m && m.user) && !m.readonly && !(page && page.viewing);

// -----------------------------------------------------------------------------
// the router: same-origin links to the app's pages change the address and the page, without
// loading a new document; back and forward return to where the page was scrolled

export const APP_PATHS = ['/', '/digest', '/profile', '/import', '/inspect', '/stats', '/agents', '/graph', '/about', '/privacy', '/imprint'];
export const location = store(new URL(window.location.href));

// shared links: ?u=<name> shows that user's tags. on someone else's page, links to the shareable
// pages keep the ?u=; on one's own with a public profile, the address carries it too, so any
// address can be copied and sent
export const SHARED_PATHS = ['/', '/digest'];
export const shared = { owner: '' };
// the list and the digest call it with their data: someone else's page, or one's own public one
export function setShared(d, m) {
    shared.owner = d.viewing ? d.owner : m.sharing ? m.user : '';
}
export function withOwner(url, owner = shared.owner) {
    if (owner && SHARED_PATHS.includes(url.pathname) && !url.searchParams.has('u')) url.searchParams.set('u', owner);
    return url;
}

history.scrollRestoration = 'manual';
let pendingScroll = null;

// own: to one of the user's own pages, also from someone else's shared one
export function navigate(href, { replace = false, own = false } = {}) {
    const url = withOwner(new URL(href, window.location.href), own ? '' : shared.owner);
    if (url.origin !== window.location.origin || !APP_PATHS.includes(url.pathname)) {
        window.location.href = url.href;
        return;
    }
    history.replaceState({ ...(history.state || {}), scroll: window.scrollY }, '');
    if (replace) history.replaceState({ scroll: 0 }, '', url.href);
    else history.pushState({ scroll: 0 }, '', url.href);
    pendingScroll = url.hash ? { hash: url.hash } : { y: 0 };
    location.set(url);
}

// the address changed without a new page (?u= added): no new entry, nothing reloads
export function setAddress(href) {
    history.replaceState(history.state, '', href);
}

window.addEventListener('popstate', e => {
    pendingScroll = { y: (e.state && e.state.scroll) || 0 };
    location.set(new URL(window.location.href));
});

// called when a page has drawn its data: scrolls to where the user was (back/forward), to the
// #anchor, or to the top of a new page
export function pageReady() {
    const s = pendingScroll;
    if (!s) return;
    pendingScroll = null;
    requestAnimationFrame(() => {
        if (s.hash) {
            const elt = document.getElementById(s.hash.slice(1));
            if (elt) { elt.scrollIntoView(); return; }
        }
        window.scrollTo(0, s.y || 0);
    });
}

const plainClick = e => !(e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey);

document.addEventListener('click', e => {
    const a = e.target.closest('a[href]');
    if (!a) return;
    const url = new URL(a.href, window.location.href);
    // links that leave the site (arXiv, code, licenses) open in a new tab, so the list stays where it was
    if (url.protocol.startsWith('http') && url.origin !== window.location.origin) {
        if (!a.target) { a.target = '_blank'; a.rel = 'noopener noreferrer'; }
        return;
    }
    if (!plainClick(e) || a.target || a.hasAttribute('download') || !APP_PATHS.includes(url.pathname)) return;
    if (url.hash && url.pathname === window.location.pathname && url.search === window.location.search) return;
    e.preventDefault();
    navigate(url.href);
});

// a chip or tab under the pointer: its page is fetched already, so the click shows it at once
export function prefetchOnHover(dataPath) {
    let timer = null;
    document.addEventListener('mouseover', e => {
        const a = e.target.closest('a[data-prefetch]');
        clearTimeout(timer);
        if (!a) return;
        timer = setTimeout(() => {
            const path = dataPath(withOwner(new URL(a.href, window.location.href)));
            if (path) load(path).catch(() => {});
        }, 80);
    });
}

// -----------------------------------------------------------------------------
// a loading indicator while the next page is on its way: the old page stays usable, and after
// a moment a bar and a line about what is happening show up

let busyTimer = null;
let busyCount = 0;

export function busy(msg) {
    busyCount++;
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

export function idle() {
    busyCount = Math.max(0, busyCount - 1);
    if (busyCount) return;
    clearTimeout(busyTimer);
    const elt = document.getElementById('busy');
    if (elt) elt.hidden = true;
}

// a form posted by the app: its fields, the busy line while it runs, and its error under it
export function useAction() {
    const [state, setState] = useState({ busy: false, error: '' });
    const run = async (fn, msg) => {
        setState({ busy: true, error: '' });
        if (msg) busy(msg);
        try {
            const r = await fn();
            setState({ busy: false, error: '' });
            return r;
        } catch (e) {
            setState({ busy: false, error: e.message });
            return undefined;
        } finally {
            if (msg) idle();
        }
    };
    return [state, run, err => setState({ busy: false, error: err })];
}

export const Err = ({ msg }) => (msg ? html`<p class="form-error">${msg}</p>` : null);

// a page whose data didn't come: the reason and a way to try again
export const LoadError = ({ error, reload }) => html`
    <div class="page"><section class="card">
        <p class="form-error">${error}</p>
        <button type="button" class="btn" onClick=${reload}>Try again</button>
    </section></div>`;
