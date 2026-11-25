// the app: header, footer and the page for the address. who is logged in comes first (/ui/me),
// each page then fetches its own data (see lib.js)
import { render } from 'preact';
import { html, useEffect, useRef, me, refreshMe, api, load, location, shared, setAddress, SHARED_PATHS, offline, unsynced, flushChanges, rememberMe, lastMe, slowCopy, SLOW_MS } from 'app/lib';
import { initAbout } from 'app/papers';
import { ListPage, listPath } from 'app/list';
import { DigestPage, InspectPage, StatsPage, TextPage } from 'app/pages';
import { ProfilePage, ImportPage, AgentsPage, HistoryPage } from 'app/account';
import { GraphPage } from 'app/graph';
import { OfflinePage } from 'app/offline';
import { ReaderPage } from 'app/reader';

const PAGES = {
    '/': ListPage,
    '/digest': DigestPage,
    '/inspect': InspectPage,
    '/stats': StatsPage,
    '/profile': ProfilePage,
    '/import': ImportPage,
    '/agents': AgentsPage,
    '/history': HistoryPage,
    '/graph': GraphPage,
    '/offline': OfflinePage,
    '/read': ReaderPage,
    '/about': () => html`<${TextPage} name="about" />`,
    '/privacy': () => html`<${TextPage} name="privacy" />`,
    '/imprint': () => html`<${TextPage} name="imprint" />`,
};

// the sticky header's height as --header-h, for things that stick just below it (the digest's
// tag headings); it changes when the nav wraps on narrow screens
const trackHeight = header => {
    if (!header) return;
    const set = () => document.documentElement.style.setProperty('--header-h', header.offsetHeight + 'px');
    set();
    if ('ResizeObserver' in window) new ResizeObserver(set).observe(header);
    else window.addEventListener('resize', set);
};

const Header = ({ m }) => {
    const n = unsynced.use();
    return html`
    <header id="header" ref=${trackHeader}>
        <a href="/" id="home-link">Arxiv <span>Resanity</span></a>
        <nav>
            <a href="/?rank=time" class="nav-secondary" title="every paper, newest first">newest</a>
            <a href="/agents" title="how to use this site with Claude and other AI agents">agents</a>
            ${m.user ? html`<a href="/?rank=unread&tags=all" title="saved papers you haven't marked as read">to read</a><a href="/digest">digest</a>` : null}
            ${m.user && !m.readonly ? html`<a href="/import" title="add papers by arXiv link">add</a>` : null}
            ${m.user && !m.readonly ? html`<a href="/offline" title="keep your papers and their PDFs on this device, for reading without a network (on a plane)">✈ offline</a>` : null}
            <a href="/profile" class=${m.user ? 'nav-user' : 'nav-login'} title=${m.readonly ? 'read-only view' : null}>
                ${m.readonly ? '👁 ' + m.user : m.user || 'log in'}</a>
            ${n ? html`<a href="/offline" class="nav-unsynced" title="changes made offline, sent when the server can be reached">⟳ ${n} to send</a>` : null}
        </nav>
    </header>`;
};

let tracked = false;
const trackHeader = elt => { if (elt && !tracked) { tracked = true; trackHeight(elt); } };

const Footer = () => html`
    <footer id="footer">
        <a href="/about">about</a> · <a href="/?rank=time">newest</a> · <a href="/?rank=random&time_filter=7">random</a> ·
        ${' '}<a href="/stats">stats</a> · <a href="/agents" title="use this site from Claude and other AI agents">AI agents</a> ·
        ${' '}<a href="https://github.com/dorian-K/arxiv-sanity-lite" title="the source code of this site">source on GitHub</a> ·
        ${' '}<a href="/privacy">privacy</a> · <a href="/imprint">imprint</a>
    </footer>`;

// right after signing up: public profile or not (the account stays private until this is answered,
// see CLAUDE.md), with the box ticked
const AskPublic = ({ m }) => {
    const ref = useRef(null);
    useEffect(() => { ref.current.showModal(); }, []);
    const submit = e => {
        e.preventDefault();
        const pub = e.target.elements.public.checked;
        api('/sharing', { public: pub ? '1' : '0' }).then(refreshMe).catch(err => alert(err.message));
    };
    return html`
    <dialog id="askpublic" aria-labelledby="askpublic-title" ref=${ref} onCancel=${e => e.preventDefault()}>
        <form onSubmit=${submit}>
            <h2 id="askpublic-title">Welcome, ${m.user}!</h2>
            <label class="check" title="Anyone with a link to your pages (it contains your username) can see your tags, saved papers, recommendations and digest, read-only; never your email address">
                <input type="checkbox" name="public" value="1" checked />
                <span><b>Public profile</b><br /><span class="muted small">Others with a link can view your tags and saved
                    papers read-only and copy a tag to their account. Never your email. You can change this any time on your profile.</span></span>
            </label>
            <button type="submit" class="btn btn-primary">Continue</button>
        </form>
    </dialog>`;
};

const App = () => {
    const m = me.use();
    const url = location.use();
    // one's own pages with a public profile carry ?u=<name>, so any address can be copied and sent
    useEffect(() => {
        if (!SHARED_PATHS.includes(url.pathname)) shared.owner = m.sharing ? m.user : '';
        if (m.sharing && SHARED_PATHS.includes(url.pathname) && !url.searchParams.has('u')) {
            const here = new URL(url.href);
            here.searchParams.set('u', m.user);
            setAddress(here.href);
        }
    }, [url, m]);
    // without a network every address shows the papers kept on the device (static/app/offline.js),
    // but the reader, which reads them from there
    const PageComponent = offline.use() && url.pathname !== '/read' ? OfflinePage : PAGES[url.pathname] || ListPage;
    const kept = slowCopy.use();
    return html`
        <${Header} m=${m} />
        ${kept ? html`<div class="slow-note" role="status">Slow connection: this is the page as you last saw it
            (${new Date(kept * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' })}).
            The current one replaces it when it arrives.</div>` : null}
        <main><${PageComponent} key=${url.pathname} /></main>
        <${Footer} />
        ${m.ask_public ? html`<${AskPublic} m=${m} />` : null}`;
};

// the first page's data is asked for together with who is logged in, not after
const here = new URL(window.location.href);
if (here.pathname === '/') load(listPath(here)).catch(() => {});
if (here.pathname === '/digest') load('/ui/digest' + here.search).catch(() => {});

const start = m => {
    me.set(m);
    initAbout(m.card_tab);
    const root = document.getElementById('app');
    root.textContent = ''; // the boot skeleton of index.html
    render(html`<${App} />`, root);
};

// a slow connection: whoever was logged in last starts the app after SLOW_MS (the page's data
// still comes from the server, or its kept copy meanwhile), corrected when /ui/me answers
let started = false;
const slowStart = setTimeout(() => {
    const m = lastMe();
    if (m && m.user) { started = true; start(m); }
}, SLOW_MS);

api('/ui/me').then(m => {
    clearTimeout(slowStart);
    rememberMe(m);
    if (started) me.set(m);
    else start(m);
    if (m.user && !m.readonly) flushChanges();
}).catch(() => {
    clearTimeout(slowStart);
    const m = lastMe();
    if (m && m.user) {
        offline.set(true);
        if (!started) start(m);
        return;
    }
    document.getElementById('app').innerHTML = '<div class="page"><section class="card"><p class="form-error">The server could not be reached. Try reloading the page.</p></section></div>';
});

// the network comes and goes (a plane): offline, every address shows the papers on the device
window.addEventListener('offline', () => { if (me.get() && me.get().user) offline.set(true); });
window.addEventListener('online', () => {
    refreshMe().then(m => {
        offline.set(false);
        if (m.user && !m.readonly) flushChanges();
    }).catch(() => {});
});

// the service worker (sw.js) answers from the device only when the server can't be reached
if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
