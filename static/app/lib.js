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

// keep: a POST that changes nothing a page shows (summaries, the card setting), so the cache stays.
// queue: false for a POST whose caller needs the server's answer (the import pages show what each
// line became), so it is never kept on the device to send later
export async function api(path, form, { keep = false, queue = true } = {}) {
    const opts = { headers: { Accept: 'application/json' } };
    const changes = form !== undefined && !keep;
    const queueable = form !== undefined && queue && QUEUEABLE.test(path);
    if (form !== undefined) {
        opts.method = 'POST';
        opts.body = new URLSearchParams(form);
    }
    if (changes) changed();
    // changes kept on the device still wait to be sent: this one goes behind them, so an older one
    // sent later can't undo it (a save kept offline, then removed online, saved again by the flush)
    if (queueable && readQueue().length) {
        queueChange(path, form);
        flushChanges();
        return 'ok, kept on this device until the earlier changes are sent';
    }
    let r;
    try {
        r = await fetch(path, opts);
    } catch (e) {
        // offline (on a plane): votes, read marks and notes wait on the device, see queueChange
        if (queueable) {
            queueChange(path, form);
            return 'ok, kept on this device until you are online';
        }
        throw new ApiError(form !== undefined ? 'Could not reach the server, so this was not saved.' : 'Could not reach the server.');
    }
    const text = await r.text();
    // again once the change is stored: a page fetched meanwhile (another worker) may predate it
    if (changes) changed();
    let data;
    try { data = JSON.parse(text); } catch (e) { data = null; }
    if (!r.ok || (data && data.error)) throw new ApiError((data && data.error) || text || r.statusText);
    if (form !== undefined && QUEUEABLE.test(path) && String(text).startsWith('ok')) paperChanged(path, form);
    return data ?? text;
}

// -----------------------------------------------------------------------------
// offline (static/app/offline.js): whether the server can't be reached, and the changes made
// meanwhile, kept in localStorage and sent in order once it can

export const offline = store(false);
// /import: a cited paper saved from the reading pack (offline.js), fetched once the server has the line
const QUEUEABLE = /^\/(?:(?:add|sub|addneg|subneg|seen|unseen|read|unread|note)\/|import$)/;
const QUEUE_KEY = 'resanity-queue';
const readQueue = () => { try { return JSON.parse(localStorage.getItem(QUEUE_KEY)) || []; } catch (e) { return []; } };
const writeQueue = q => { try { localStorage.setItem(QUEUE_KEY, JSON.stringify(q)); } catch (e) { /* full or blocked */ } };
export const unsynced = store(readQueue().length);

// a change to a paper (vote, read mark, note), sent or kept: the reading pack on the device follows it
const paperListeners = new Set();
export const onPaperChange = f => paperListeners.add(f);
const paperChanged = (path, form) => paperListeners.forEach(f => { try { f(path, form); } catch (e) { /* the pack is a copy */ } });

function queueChange(path, form) {
    const q = readQueue();
    // a FormData spreads to {}: its entries are what is sent
    q.push([path, form instanceof FormData ? Object.fromEntries(form) : { ...form }]);
    writeQueue(q);
    unsynced.set(q.length);
    paperChanged(path, form);
}

// sends what was kept, oldest first; stops at the first that doesn't reach the server. an answer
// that isn't "ok" (a tag deleted meanwhile) won't get better by sending it again, so it is dropped,
// unless the session has ended: then it waits for the next login
let flushing = false, retry = null;
export async function flushChanges() {
    if (flushing || !readQueue().length) return;
    flushing = true;
    // one tab at a time: two tabs sending the same queue sent every change twice, in any order
    if (navigator.locks) await navigator.locks.request('resanity-flush', sendQueue);
    else await sendQueue();
}
async function sendQueue() {
    try {
        for (;;) {
            const q = readQueue();
            if (!q.length) break;
            const [path, form] = q[0];
            let r, text;
            try {
                r = await fetch(path, { method: 'POST', headers: { Accept: 'application/json' }, body: new URLSearchParams(form) });
                text = await r.text();
            } catch (e) {
                break;
            }
            if (r.status === 429) { // rate limited: again when it says, not only at the next page load
                clearTimeout(retry);
                retry = setTimeout(flushChanges, Math.min(120, Number(r.headers.get('Retry-After')) || 30) * 1000);
                break;
            }
            if (r.status === 403 || r.status >= 500 || /not logged in/.test(text)) break;
            writeQueue(readQueue().slice(1));
        }
    } finally {
        flushing = false;
        unsynced.set(readQueue().length);
        changed();
    }
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
    const [state, setState] = useState({ data: null, error: null, path: null, kept: 0 });
    const [round, setRound] = useState(0);
    useEffect(() => {
        let live = true, fresh = false, ok = false, shownKept = false;
        if (path === null) return undefined;
        busy(busyMsg || 'Loading…');
        // a slow connection: the copy kept on the device shows meanwhile, marked as such
        const slow = setTimeout(() => keptCopy(path).then(k => {
            if (!k || !live || fresh) return;
            setState({ data: k.data, error: null, path, kept: k.time });
            slowCopy.set(k.time);
            shownKept = true;
            pageReady();
        }), SLOW_MS);
        load(path).then(
            data => { fresh = ok = true; if (live) setState({ data, error: null, path, kept: 0 }); keepCopy(path, data); },
            e => { fresh = true; if (live) setState(s => ({ ...s, error: e.message, path })); },
        ).finally(() => {
            clearTimeout(slow);
            idle();
            // a failed load leaves the kept copy on screen: its note stays with it
            if (live) { if (ok || !shownKept) slowCopy.set(0); pageReady(); }
        });
        return () => { live = false; clearTimeout(slow); slowCopy.set(0); };
    }, [path, round]);
    return {
        data: state.data,
        error: state.path === path ? state.error : null,
        loading: state.path !== path,
        kept: state.path === path ? state.kept : 0, // the data is the copy on the device, from then (s)
        dataPath: state.path, // what data shows: the previous page's while the next one loads
        reload: () => { cache.delete(path); setRound(n => n + 1); },
    };
}

// the last pages' data, kept on the device for a slow connection: shown after SLOW_MS without an
// answer, until the server's arrives (app.js says so above the page). only the main lists and the
// digest; deleted on logout (forgetDevice)
export const SLOW_MS = 2500;
const KEPT = 'data-v1';
const KEPT_MAX = 20;
const KEEPABLE = /^\/ui\/(list|digest)(\?|$)/;
export const slowCopy = store(0); // when the copy on screen was made (s), 0: the server's data
// never a real address (sw.js and the server don't see it), per user
const keptUrl = path => '/kept/' + encodeURIComponent((me.get() && me.get().user) || '') + path;

async function keptCopy(path) {
    if (!KEEPABLE.test(path) || typeof caches === 'undefined') return null;
    try {
        const r = await caches.match(keptUrl(path), { cacheName: KEPT });
        return r ? await r.json() : null;
    } catch (e) {
        return null;
    }
}

async function keepCopy(path, data) {
    if (!KEEPABLE.test(path) || typeof caches === 'undefined' || !(me.get() && me.get().user)) return;
    try {
        const c = await caches.open(KEPT);
        await c.delete(keptUrl(path)); // to the end of the line
        await c.put(keptUrl(path), new Response(JSON.stringify({ time: Date.now() / 1000, data })));
        const keys = await c.keys();
        for (const k of keys.slice(0, Math.max(0, keys.length - KEPT_MAX))) await c.delete(k);
    } catch (e) { /* not kept */ }
}

// -----------------------------------------------------------------------------
// who is looking: their name, whether read-only, the site's switches (/ui/me)

export const me = store(null);
export const refreshMe = () => api('/ui/me').then(m => { changed(); rememberMe(m); me.set(m); return m; });
// who was logged in last, for opening the site without a network (offline.js)
const ME_KEY = 'resanity-me';
export const rememberMe = m => {
    claimDevice(m);
    try { localStorage.setItem(ME_KEY, JSON.stringify(m)); } catch (e) { /* not remembered */ }
};
// whose the reading pack, the queued changes and the kept lists on this device are. someone else logging
// in (a session that ended without a logout on a shared computer) must neither see them nor have the
// changes sent into their account: they are deleted, as on logout. the same user again keeps them
const OWNER_KEY = 'resanity-owner';
function claimDevice(m) {
    if (!m || !m.user) return; // not logged in (a session that ran out): kept for when they are back
    try {
        let owner = localStorage.getItem(OWNER_KEY);
        if (owner === null) owner = (lastMe() || {}).user || null; // from before owners were written down
        if (owner && owner !== m.user) forgetDevice();
        localStorage.setItem(OWNER_KEY, m.user);
    } catch (e) { /* no localStorage: nothing kept either */ }
}
export const lastMe = () => { try { return JSON.parse(localStorage.getItem(ME_KEY)); } catch (e) { return null; } };
// logging out: nothing of the account stays on the device (a shared computer)
export const forgetDevice = () => {
    try { [ME_KEY, QUEUE_KEY, OWNER_KEY].forEach(k => localStorage.removeItem(k)); } catch (e) { /* nothing there */ }
    unsynced.set(0);
    return typeof caches !== 'undefined'
        ? Promise.all(['pack-v1', KEPT].map(n => caches.delete(n))).catch(() => {})
        : Promise.resolve();
};
// a button that copies a link (made absolute) to the clipboard, and says so for a moment
export const CopyLink = ({ href, label = '🔗 Copy link', title = 'copy a link to this', className = 'btn btn-small' }) => {
    const [copied, setCopied] = useState(false);
    const copy = () => {
        const url = new URL(href, window.location.href).href;
        const done = () => { setCopied(true); setTimeout(() => setCopied(false), 2000); };
        if (navigator.clipboard) navigator.clipboard.writeText(url).then(done, () => window.prompt('Copy this link:', url));
        else window.prompt('Copy this link:', url);
    };
    return html`<button type="button" class=${className} onClick=${copy} title=${title}>${copied ? '✓ Link copied' : label}</button>`;
};
export const canEdit = (m, page) => Boolean(m && m.user) && !m.readonly && !(page && page.viewing);

// -----------------------------------------------------------------------------
// the router: same-origin links to the app's pages change the address and the page, without
// loading a new document; back and forward return to where the page was scrolled

export const APP_PATHS = ['/', '/digest', '/profile', '/import', '/inspect', '/stats', '/agents', '/graph', '/offline', '/read', '/about', '/privacy', '/imprint'];
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
// the first page: its #anchor (a link in a mail), or where a reload left it
let pendingScroll = window.location.hash ? { hash: window.location.hash } : { y: (history.state && history.state.scroll) || 0 };
window.addEventListener('pagehide', () => history.replaceState({ ...(history.state || {}), scroll: window.scrollY }, ''));

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
