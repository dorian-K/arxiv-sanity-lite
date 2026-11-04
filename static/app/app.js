// the app: header, footer and the page for the address. who is logged in comes first (/ui/me),
// each page then fetches its own data (see lib.js)
import { render } from 'preact';
import { html, useEffect, useRef, me, refreshMe, api, location, shared, setAddress, SHARED_PATHS } from 'app/lib';
import { initAbout } from 'app/papers';
import { ListPage } from 'app/list';
import { DigestPage, InspectPage, StatsPage, TextPage } from 'app/pages';
import { ProfilePage, ImportPage, AgentsPage } from 'app/account';
import { GraphPage } from 'app/graph';

const PAGES = {
    '/': ListPage,
    '/digest': DigestPage,
    '/inspect': InspectPage,
    '/stats': StatsPage,
    '/profile': ProfilePage,
    '/import': ImportPage,
    '/agents': AgentsPage,
    '/graph': GraphPage,
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

const Header = ({ m }) => html`
    <header id="header" ref=${trackHeader}>
        <a href="/" id="home-link">Arxiv <span>Resanity</span></a>
        <nav>
            <a href="/?rank=time" class="nav-secondary" title="every paper, newest first">newest</a>
            <a href="/agents" title="how to use this site with Claude and other AI agents">agents</a>
            ${m.user ? html`<a href="/?rank=unread&tags=all" title="saved papers you haven't marked as read">to read</a><a href="/digest">digest</a>` : null}
            ${m.user && !m.readonly ? html`<a href="/import" title="add papers by arXiv link">add</a>` : null}
            <a href="/profile" class=${m.user ? 'nav-user' : 'nav-login'} title=${m.readonly ? 'read-only view' : null}>
                ${m.readonly ? '👁 ' + m.user : m.user || 'log in'}</a>
        </nav>
    </header>`;

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
    const PageComponent = PAGES[url.pathname] || ListPage;
    return html`
        <${Header} m=${m} />
        <main><${PageComponent} key=${url.pathname} /></main>
        <${Footer} />
        ${m.ask_public ? html`<${AskPublic} m=${m} />` : null}`;
};

refreshMe().then(m => {
    initAbout(m.card_tab);
    render(html`<${App} />`, document.getElementById('app'));
}).catch(() => {
    document.getElementById('app').innerHTML = '<div class="page"><section class="card"><p class="form-error">The server could not be reached. Try reloading the page.</p></section></div>';
});
