// the pages about the user's account: log in and settings (profile), adding papers, and access
// tokens for AI agents. their forms post to the server's endpoints and show the answer in place
import { html, useState, me, refreshMe, api, useData, navigate, Err, useAction, LoadError } from 'app/lib';
import { about, setAboutDefault } from 'app/papers';

const fmtDate = t => (t ? new Date(t * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'never');
// an input of a form kept in one state object. functional update: a password manager fills
// several fields at once, before any re-render
const field = (state, set) => e => {
    const { name, value } = e.target;
    set(s => ({ ...s, [name]: value }));
};

// -----------------------------------------------------------------------------
// logging in (and signing up: the first login with a new name)

const LoginForm = ({ user }) => {
    const [f, setF] = useState({ username: user || '', password: '' });
    const [state, run] = useAction();
    const submit = e => {
        e.preventDefault();
        // what the form holds right now (a password manager may have just filled it)
        const form = new FormData(e.target);
        if (user) form.set('username', user);
        run(() => api('/login', form), 'Logging in…').then(r => r && refreshMe().then(() => navigate(r.next || '/', { own: true })));
    };
    return html`
    <form id="profile-login-form" onSubmit=${submit}>
        <${Err} msg=${state.error} />
        ${user ? null : html`
        <div class="form-group">
            <label for="username">Username</label>
            <input class="form-control" type="text" id="username" name="username" autocomplete="username"
                   autocapitalize="none" spellcheck="false" maxLength="40" required value=${f.username} onInput=${field(f, setF)} />
        </div>`}
        <div class="form-group">
            <label for="password">${user ? 'Password for ' + user : 'Password'}</label>
            <input class="form-control" type="password" id="password" name="password" autocomplete="current-password"
                   required=${Boolean(user)} value=${f.password} onInput=${field(f, setF)} />
        </div>
        <button type="submit" class="btn btn-primary" disabled=${state.busy}>${user ? 'Log in' : 'Log in / sign up'}</button>
        ${user ? null : html`<p class="muted small">Leave the password empty to view a public profile read-only.
            How we handle your data: <a href="/privacy">privacy policy</a>.</p>`}
    </form>`;
};

const logout = () => api('/logout', {}).then(refreshMe);

// -----------------------------------------------------------------------------
// the profile: settings, digest, data

const Sharing = ({ p, user, reload }) => {
    const [state, run] = useAction();
    const toggle = e => run(() => api('/sharing', { public: e.target.checked ? '1' : '0' })).then(r => r && refreshMe().then(reload));
    return html`
    <section class="card" id="sharing">
        <h1>Public profile</h1>
        <label class="check">
            <input type="checkbox" checked=${p.public} disabled=${state.busy} onChange=${toggle} />
            <span>Public profile: anyone with a link can view my tags read-only</span>
        </label>
        <${Err} msg=${state.error} />
        <p class="muted">Then every page address you see has <code>?u=${user}</code> in it, and "🔗 Share" copies it:
            others see your tags, recommendations and digest there (never your email), can't change anything,
            and can copy a tag to their own account.
            ${' '}${p.public ? 'Turning this off makes it private and ends their access right away.' : 'Private now: nobody but you sees your tags.'}</p>
        ${p.public ? html`<p>Your page: <a href=${'/?u=' + encodeURIComponent(user)}>/?u=${user}</a></p>` : null}
    </section>`;
};

const CardTab = () => {
    const tab = about.use();
    return html`
    <section class="card" id="cards">
        <h1>Paper cards</h1>
        <p class="muted">Cards with an AI summary have two tabs, Summary and Abstract. Which one they open on:</p>
        <label class="check" title="problem, approach, result and a caveat, written by an AI model">
            <input type="radio" name="tab" value="summary" checked=${tab !== 'abstract'} onChange=${() => setAboutDefault('summary')} />
            <span>The AI summary</span>
        </label>
        <label class="check" title="the paper's abstract, as the authors wrote it">
            <input type="radio" name="tab" value="abstract" checked=${tab === 'abstract'} onChange=${() => setAboutDefault('abstract')} />
            <span>The abstract</span>
        </label>
    </section>`;
};

const Email = ({ p, reload }) => {
    const [email, setEmail] = useState('');
    const [state, run] = useAction();
    const send = value => run(() => api('/register_email', { email: value }), 'Sending…').then(r => { if (r) { setEmail(''); reload(); } });
    return html`
    <${Err} msg=${state.error} />
    <p class="muted">
        Every Sunday, an email with the best new papers for each of your tags, linking to your
        ${' '}<a href="/digest">digest page</a> where you can rate them. Your address is used for nothing else,
        and every digest has an unsubscribe link (<a href="/privacy">privacy</a>).
    </p>
    ${p.email ? html`<p>Digests go to <b>${p.email}</b> (confirmed ${p.email_confirmed}).</p>` : null}
    ${p.pending_email ? html`<p>We sent a confirmation link to <b>${p.pending_email}</b>. Nothing else is sent there until
        you click it (it works for 3 days).</p>` : null}
    <form id="email-form" onSubmit=${e => { e.preventDefault(); send(email); }}>
        <div class="form-group">
            <input type="email" class="form-control" name="email" maxLength="254" autocomplete="email" required
                   placeholder=${p.email ? 'New email address' : 'Email address'} value=${email} onInput=${e => setEmail(e.target.value)} />
        </div>
        <button type="submit" class="btn btn-primary" disabled=${state.busy}>${p.email ? 'Change address' : 'Send confirmation link'}</button>
    </form>
    ${p.email || p.pending_email ? html`
    <button type="button" class="btn" style=${{ marginTop: '10px' }} disabled=${state.busy}
            onClick=${() => send('')}>Stop the digest and delete my address</button>` : null}`;
};

const DigestSizes = ({ p, reload }) => {
    const [days, setDays] = useState(String(p.digest_days));
    const [sizes, setSizes] = useState(() => Object.fromEntries(p.digest_tags.map(t => [t.name, String(t.n)])));
    const [saved, setSaved] = useState(false);
    const [state, run] = useAction();
    const submit = e => {
        e.preventDefault();
        const form = new URLSearchParams({ days });
        for (const [tag, n] of Object.entries(sizes)) { form.append('tag', tag); form.append('n', n); }
        run(() => api('/digest_sizes', form)).then(r => { if (r) { setSaved(true); reload(); } });
    };
    return html`
    <h2 id="digest-sizes">What the digest shows</h2>
    <p class="muted">It comes every week; the window says how far back it looks for papers (say 14 days, to
        see two weeks of papers in each mail). Papers per tag: 0 leaves the tag out. Tags an AI agent made
        (<code>agent:…</code>) start at 0.</p>
    ${saved ? html`<p class="form-ok">Saved.</p>` : null}
    <${Err} msg=${state.error} />
    <form class="digest-sizes" onSubmit=${submit} onInput=${() => setSaved(false)}>
        <label title=${'the digest takes papers that came out in the last this many days (1 to ' + p.max_digest_days + ')'}>
            <span>Window, in days</span>
            <input class="form-control" type="number" name="days" min="1" max=${p.max_digest_days} value=${days}
                   onInput=${e => setDays(e.target.value)} required />
        </label>
        <span class="muted">Papers per tag:</span>
        ${p.digest_tags.map(t => html`
        <label key=${t.name}>
            <span>${t.name}</span>
            <input class="form-control" type="number" min="0" max=${p.max_per_tag} value=${sizes[t.name]} required
                   onInput=${e => { const v = e.target.value; setSizes(s => ({ ...s, [t.name]: v })); }} />
        </label>`)}
        <button type="submit" class="btn" disabled=${state.busy}>Save</button>
    </form>`;
};

const DeleteAccount = ({ onDeleted }) => {
    const [password, setPassword] = useState('');
    const [state, run] = useAction();
    const submit = e => {
        e.preventDefault();
        run(() => api('/delete_account', { password })).then(r => r && refreshMe().then(onDeleted));
    };
    return html`
    <details class="danger">
        <summary>Delete my account</summary>
        <form onSubmit=${submit}>
            <p>This erases your account with all its tags, votes, digests and your email address, for good.</p>
            <${Err} msg=${state.error} />
            <div class="form-group">
                <label for="del-password">Your password, to confirm</label>
                <input class="form-control" type="password" id="del-password" autocomplete="current-password" required
                       value=${password} onInput=${e => setPassword(e.target.value)} />
            </div>
            <button type="submit" class="btn btn-danger" disabled=${state.busy}>Delete my account</button>
        </form>
    </details>`;
};

export const ProfilePage = () => {
    const m = me.use();
    const [deleted, setDeleted] = useState(false);
    // the settings of a logged-in user; nothing to fetch for the login form
    const { data: p, error, reload } = useData(m.user && !m.readonly ? '/ui/profile?' + m.user : null);
    if (deleted) {
        return html`<div class="page"><section class="card">
            <h1>Account deleted</h1>
            <p>Your account and everything stored with it (tags, votes, digests, email address) are gone.
                The daily backup is overwritten within a day, after which no copy is left.</p>
        </section></div>`;
    }
    if (m.readonly) {
        return html`<div id="profilewrap" class="page"><section class="card">
            <h1>Viewing ${m.user}, read-only</h1>
            <p class="muted">${m.user} has a public profile, so you can browse their tags, recommendations and digest.
                To change anything, log in with the password.</p>
            <${LoginForm} user=${m.user} key=${m.user} />
            <button type="button" class="btn" style=${{ marginTop: '10px' }} onClick=${logout}>Stop viewing</button>
        </section></div>`;
    }
    if (!m.user) {
        return html`<div id="profilewrap" class="page"><section class="card">
            <h1>Log in</h1>
            <p class="muted">${m.allow_signup ? "New here? Pick a username and a password (at least 8 characters) and you're signed up."
                : 'Signups are closed on this server.'}</p>
            <${LoginForm} />
        </section></div>`;
    }
    if (!p) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    return html`
    <div id="profilewrap" class="page">
        <section class="card">
            <h1>Logged in as ${m.user}</h1>
            <button type="button" class="btn" onClick=${logout}>Log out</button>
        </section>
        <${Sharing} p=${p} user=${m.user} reload=${reload} />
        ${m.summaries_on ? html`<${CardTab} />` : null}
        <section class="card">
            <h1>Weekly digest</h1>
            <${Email} p=${p} reload=${reload} />
            ${p.digest_tags.length ? html`<${DigestSizes} p=${p} reload=${reload} />` : null}
        </section>
        <section class="card">
            <h1>AI agents</h1>
            <p class="muted">Let Claude Code or another agent search this site and read (or organize) your tags,
                with an access token you can revoke any time.</p>
            <p><a class="btn" href="/agents">Access tokens for agents</a></p>
        </section>
        <section class="card">
            <h1>Your data</h1>
            <p class="muted">Everything stored about you: your account, tags, votes, digests and email address
                (<a href="/privacy">privacy policy</a>).</p>
            <p><a class="btn" href="/export" download>Download my data</a></p>
            <${DeleteAccount} onDeleted=${() => setDeleted(true)} />
        </section>
    </div>`;
};

// -----------------------------------------------------------------------------
// adding papers by arXiv link or id, into a tag

export const ImportPage = () => {
    const m = me.use();
    const { data: d, error, reload } = useData('/ui/import?' + (m.user || ''));
    const [f, setF] = useState({ text: '', tag: '' });
    const [res, setRes] = useState(null);
    const [state, run] = useAction();
    if (!d) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    const submit = e => {
        e.preventDefault();
        run(() => api('/import', new FormData(e.target)), "Adding papers… ones that aren't here yet are fetched from arXiv, which takes a few seconds per 100.")
            .then(r => { if (r) { setRes(r); reload(); } });
    };
    const q = res && encodeURIComponent(res.tag);
    return html`
    <div id="importwrap" class="page">
        <section class="card">
            <h1>Add papers</h1>
            ${!m.user || m.readonly ? html`<p class="muted">Please <a href="/profile">log in</a> to add papers.</p>` : html`
            <p class="muted">
                One arXiv link or id per line, up to ${d.max_lines} at a time: arxiv.org/abs or /pdf links,
                Hugging Face or alphaXiv paper pages, DOIs, or bare ids like 2401.12345. Papers of any year
                and category work; ones that aren't here yet are fetched from arXiv (up to ${d.limit} a day).
            </p>
            <${Err} msg=${state.error} />
            <form id="import-form" onSubmit=${submit}>
                <div class="form-group">
                    <textarea class="form-control" name="text" rows="8" required spellcheck="false"
                              placeholder=${'https://arxiv.org/abs/1706.03762\n2401.12345'} value=${f.text} onInput=${field(f, setF)} />
                </div>
                <div class="form-group">
                    <label for="import-tag">Save them to tag: pick one of yours or type a new name</label>
                    <input class="form-control" type="text" id="import-tag" name="tag" maxLength="100" value=${f.tag} onInput=${field(f, setF)}
                           autocapitalize="none" spellcheck="false" autocomplete="off" list="import-tags" required />
                    <datalist id="import-tags">${d.user_tags.map(t => html`<option key=${t} value=${t} />`)}</datalist>
                </div>
                <button type="submit" class="btn btn-primary" disabled=${state.busy}>Add papers</button>
            </form>`}
        </section>
        ${res && res.results.length ? html`
        <section class="card">
            <h1>Results</h1>
            <table class="stattable import-results">
                ${res.results.map((r, i) => html`<tr key=${i}>
                    <td>${r.pid ? html`<a href=${'/?rank=pid&pid=' + r.pid}>${r.title || r.pid}</a>` : html`<code>${r.line}</code>`}</td>
                    <td class=${r.pid ? 'muted' : 'form-error'}>
                        ${r.status}${r.pid && !r.indexed ? ', used for recommendations after the next update' : ''}</td>
                </tr>`)}
            </table>
            <p><a href=${'/?rank=saved&tags=' + q}>See all papers in ${res.tag}</a> ·
                ${' '}<a href=${'/?rank=tags&tags=' + q + '&skip_have=yes'}>Recommendations for ${res.tag}</a></p>
        </section>` : null}
    </div>`;
};

// -----------------------------------------------------------------------------
// AI agents: how to connect one, and the user's access tokens

const ACCESS = [
    ['read', 'it can read your tags, lists, notes and recommendations, and search and rank papers, but change nothing',
        html`<b>Nothing</b>: read only.`],
    ['agent', "your own tags stay untouched: every tag it saves to, removes from or marks papers for is named agent:..., the prefix added automatically. It can't change read marks or notes",
        html`<b>Only its own tags</b>: tags named <code>agent:…</code> (it can't touch your other tags, read marks or notes).`],
    ['write', 'it can save papers to any of your tags, mark them as not for a tag or as read, and write notes; it can never delete a tag',
        html`<b>Everything</b>: any of your tags, read marks and notes (it can never delete a tag).`],
];
const NEEDS = { public: 'nothing', user: 'a token', agent: 'a token that may change its own tags (or everything)', write: 'a token that may change everything' };

const NewToken = ({ token, base }) => html`
    <section class="card" id="newtoken">
        <h1>Your new token</h1>
        <p class="form-ok">Copy it now: it is shown only this once (only a hash of it is stored).</p>
        <pre class="codeblock">${token}</pre>
        <p>For Claude Code:</p>
        <pre class="codeblock">claude mcp add --transport http resanity ${base}/mcp --header "Authorization: Bearer ${token}"</pre>
        <p>Other MCP clients: the server is <code>${base}/mcp</code> (streamable HTTP), with the header
            ${' '}<code>Authorization: Bearer ${token}</code>. Or without MCP:</p>
        <pre class="codeblock">curl -H "Authorization: Bearer ${token}" "${base}/api/recommend?days=7&limit=5&abstracts=false"</pre>
    </section>`;

const Tokens = ({ d, reload, onNew }) => {
    const [f, setF] = useState({ name: '', access: 'agent' });
    const [state, run] = useAction();
    const make = e => {
        e.preventDefault();
        run(() => api('/agents/token', new FormData(e.target))).then(r => { if (r) { onNew(r.token); setF({ name: '', access: 'agent' }); reload(); window.scrollTo(0, 0); } });
    };
    const revoke = id => run(() => api('/agents/revoke', { id })).then(r => r && reload());
    return html`
    <section class="card" id="tokens">
        <h1>Access tokens</h1>
        <${Err} msg=${state.error} />
        ${!d.can_make ? html`<p class="muted"><a href="/profile">Log in</a> to make a token for your account.</p>` : html`
        ${d.tokens.length ? html`
        <div class="tablescroll"><table class="perftable">
            <tr><th>Name</th><th>Access</th><th>Made</th><th>Last used</th><th></th></tr>
            ${d.tokens.map(t => html`<tr key=${t.id}>
                <td>${t.name}</td><td>${d.scope_labels[t.scope]}</td><td>${fmtDate(t.created)}</td><td>${fmtDate(t.last_used)}</td>
                <td><button type="button" class="btn btn-small" title="the token stops working at once" onClick=${() => revoke(t.id)}>Revoke</button></td>
            </tr>`)}
        </table></div>` : html`<p class="muted">No tokens yet.</p>`}
        <form class="tokenform" onSubmit=${make}>
            <div class="form-group">
                <label for="token-name">Name, to tell your tokens apart</label>
                <input class="form-control" id="token-name" name="name" maxLength="60" placeholder="Claude Code on my laptop"
                       value=${f.name} onInput=${field(f, setF)} />
            </div>
            <fieldset class="access-choice">
                <legend>What may it change?</legend>
                ${ACCESS.map(([v, title, label]) => html`
                <label class="check" title=${title} key=${v}>
                    <input type="radio" name="access" value=${v} checked=${f.access === v} onChange=${field(f, setF)} />
                    <span>${label}</span>
                </label>`)}
            </fieldset>
            <button type="submit" class="btn btn-primary" disabled=${state.busy}>Make a token</button>
        </form>`}
    </section>`;
};

export const AgentsPage = () => {
    const m = me.use();
    const { data: d, error, reload } = useData('/ui/agents?' + (m.user || ''));
    const [token, setToken] = useState('');
    if (!d) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    const base = d.base;
    return html`
    <div id="agentswrap" class="page">
        ${token ? html`<${NewToken} token=${token} base=${base} />` : null}
        <section class="card prose">
            <h1>Use Arxiv Resanity with AI agents</h1>
            <div>
                An AI agent such as Claude can use this site for you: search it, find related work from a few
                example papers, and, if you let it, read and organize your tags, reading list and notes. You
                ask in plain language; the agent calls the site's tools and reports back.
            </div>
            <div>
                Its strongest tool is <b>rank_like</b>. Give it a few papers you know are relevant, and it
                ranks every paper here by them, the way your tags rank new papers. That finds work that shares
                no keywords with what you would have searched for. The agent can repeat it, adding good hits
                and steering away from off-topic ones.
            </div>

            <h2>Connect</h2>
            <p><b>claude.ai and the Claude apps.</b> Settings → Connectors → Add custom connector, with the address</p>
            <pre class="codeblock">${base}/mcp/me</pre>
            <p>Claude sends you to this site to log in and choose what it may do (read only, only its own <code>agent:</code> tags,
                or everything). The connection then appears under <a href="#tokens">access tokens</a> below, where you
                can end it any time.</p>
            <p><b>Claude Code.</b> For searching and ranking, no account is needed:</p>
            <pre class="codeblock">claude mcp add --transport http resanity ${base}/mcp</pre>
            <p>For your own tags, make an access token below and add it as a header; the page then shows the
                exact command.</p>
            <p><b>Other agents and scripts.</b> Any MCP client can use <code>${base}/mcp</code> (streamable
                HTTP) with the header <code>Authorization: Bearer ${'<token>'}</code>. Without MCP, the same tools
                are plain JSON under <code>${base}/api</code>, and <a href="/llms.txt">llms.txt</a> describes
                them for agents. For example:</p>
            <pre class="codeblock">curl "${base}/api/rank_like?like=2006.11239,2112.10752&days=90&limit=5&abstracts=false"</pre>

            <h2>Things to ask</h2>
            <ul>
                <li>"These papers are what I'm working on: 2006.11239, 2011.13456, 2112.10752. Find the most
                    relevant papers of the last three months and tell me what's new in them."</li>
                <li>"Go through my reading list for the tag <i>diffusion</i>, pick the five that matter most for
                    my project, and write a note on each saying why."</li>
                <li>"Look at this week's recommendations and save the ones about video generation to a new tag
                    ${' '}<i>video</i>."</li>
                <li>"Which papers do my saved papers in <i>robotics</i> cite most that I haven't saved yet?"</li>
                <li>"I've read 2401.12345 and 2402.23456, mark them as read."</li>
            </ul>

            <h2>Good to know</h2>
            <ul>
                <li>A token acts as you. It is read-only unless you allow changes: either only to tags named
                    ${' '}<code>agent:…</code> (your own tags stay yours), or to everything; even then it can never
                    delete a tag. Keep it like a password, and revoke it when you no longer need it.</li>
                <li>What the agent reads (your tags, notes, the papers) goes to whoever runs it, for Claude that
                    is Anthropic (<a href="/privacy">privacy</a>).</li>
                <li>The site covers papers in ${d.categories}. With changes allowed, an agent can add any other arXiv paper by id.</li>
                <li>References and "cited by" are only known for papers this site has processed: the ones its
                    users saved.</li>
                <li>Agents have rate limits like everyone; without an account, they can rank by example papers
                    30 times a minute.</li>
            </ul>
        </section>
        <${Tokens} d=${d} reload=${reload} onNew=${setToken} />
        <section class="card">
            <h1>Tools</h1>
            <div class="tablescroll">
            <table class="perftable">
                <tr><th>Tool</th><th>Needs</th><th>What it does</th></tr>
                ${d.tools.map(([name, access, desc]) => html`<tr key=${name}><td><code>${name}</code></td><td>${NEEDS[access]}</td><td>${desc}</td></tr>`)}
            </table>
            </div>
        </section>
    </div>`;
};
