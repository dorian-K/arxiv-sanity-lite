// the main page, / and its ?rank=...: a list of papers (recommendations, a tag's saved papers,
// search, newest, similar to a paper), with the user's tags, the tabs of a tag and the
// ranking's settings. every choice changes the address and fetches /ui/list with it
import { html, setShared, useState, useEffect, me, api, useData, navigate, location, tagRequest, canEdit, Err, useAction, prefetchOnHover } from 'app/lib';
import { Page, PaperList, Skeleton, ViewSwitch, WordList, TagCheck, ViewBanner, tagUrl, FOR_YOU_DAYS, withMadeTags } from 'app/papers';

// a page address -> the address of its data
export const listPath = url => '/ui/list' + url.search;
// on hover, only lists that cost nothing to make: a hovered Recommended chip would train a model
// (and use up the user's rate limit) for a page they may never open
const CHEAP = ['saved', 'unread', 'down', 'cited', 'time'];
prefetchOnHover(url => (url.pathname === '/' && CHEAP.includes(url.searchParams.get('rank')) && !url.searchParams.get('q') ? listPath(url) : null));

const TABS = ['saved', 'unread', 'down', 'cited'];
const SLOW_RANKS = ['tags', 'pid']; // they train a model: a skeleton of the list meanwhile

// a chip in the tag row switches the tag and keeps what is shown: the same tab, the same time window
const chipUrl = (g, tag) => (TABS.includes(g.rank)
    ? '/?rank=' + g.rank + '&tags=' + encodeURIComponent(tag)
    : tagUrl(tag, g.rank === 'tags' ? g.time_filter : FOR_YOU_DAYS)) + (g.published ? '&published=yes' : '');

const Tag = ({ tag: t, g }) => {
    const active = ['tags', ...TABS].includes(g.rank) && g.tags === t.name;
    return html`
        <span class=${'chip chip-tag chip-count' + (t.name === 'all' ? ' rel_utag_all' : '') + (active ? ' active' : '')}>
            <a href=${chipUrl(g, t.name)} data-prefetch>${t.name === 'all' ? 'All tags' : t.name}${t.n !== undefined ? html` <small>${t.n}</small>` : null}</a>
        </span>`;
};

const TagList = ({ tags, g, d }) => html`
    <section id="tagwrap"><div id="tagList">
        <span class="taglabel">${d.viewing ? '@' + d.owner + "'s tags" : 'Your tags'}</span>
        ${tags.map(t => html`<${Tag} key=${t.name} tag=${t} g=${g} />`)}
    </div></section>`;

// the page's address with one setting changed: the bare / with its defaults spells them out first,
// so only the one choice changes
const withSetting = (g, set) => {
    const q = new URLSearchParams(window.location.search);
    if (!q.has('rank')) {
        q.set('rank', g.rank); q.set('tags', g.tags); q.set('skip_have', g.skip_have);
        if (g.time_filter) q.set('time_filter', g.time_filter);
    }
    set(q);
    q.delete('page_number');
    return '/?' + q.toString();
};

const PubFilter = ({ g }) => html`
    <label class="pubfilter" title="Only papers whose authors say where they were accepted or published (a conference, journal or workshop), taken from the arXiv comment or journal reference. Not verified: authors fill these in themselves, and often only later, so recent papers and some published ones won't show up">
        <input type="checkbox" checked=${g.published}
               onChange=${e => navigate(withSetting(g, q => (e.target.checked ? q.set('published', 'yes') : q.delete('published'))))} /> accepted / published only
    </label>`;

const WINDOWS = [['7', 'last 7 days'], ['30', 'last 30 days'], ['365', 'last year'], ['', 'any time']];

// the page of one tag (or of all): what (tabs), over how long, and how it is shown
const TagHead = ({ d, g, editable }) => {
    const all = g.tags === 'all';
    const where = all ? '' : ' to ' + g.tags;
    const q = '&tags=' + encodeURIComponent(g.tags) + (g.published ? '&published=yes' : '');
    const tab = (rank, label, title, extra = '') => html`
        <a href=${'/?rank=' + rank + q + extra} class=${g.rank === rank ? 'active' : ''} title=${title} data-prefetch>${label}</a>`;
    const deleteTag = () => {
        if (!confirm("Delete the tag '" + g.tags + "'? Its saved papers and downvotes are forgotten (the papers stay in the database).")) return;
        tagRequest('/del/' + encodeURIComponent(g.tags)).then(() => navigate('/')).catch(() => {});
    };
    const windows = WINDOWS.some(([v]) => v === g.time_filter) ? WINDOWS : [...WINDOWS, [g.time_filter, 'last ' + g.time_filter + ' days']];
    const mine = editable;
    return html`
    <section id="taghead">
        <h1>${d.viewing ? html`<span class="muted">@${d.owner} /</span> ` : null}${all ? 'All tags' : g.tags}
            ${!all && mine ? html`<button type="button" class="linkbtn deltag" onClick=${deleteTag}>delete tag</button>` : null}</h1>
        <div class="tabrow">
            <nav class="tabs" aria-label=${g.tags}>
                ${tab('tags', 'Recommended', 'new papers that match ' + (all ? 'your tags' : 'this tag') + ', best first',
                      '&time_filter=' + (g.time_filter || d.for_you_days) + '&skip_have=yes' + (g.mix === 'turns' ? '&mix=turns' : ''))}
                ${tab('saved', 'Saved (' + d.n_saved + ')', 'the papers you saved' + where)}
                ${tab('unread', 'To read (' + d.n_unread + ')', "the papers you saved" + where + " that you haven't marked as read")}
                ${tab('down', 'Not for ' + (all ? 'a tag' : 'this tag') + ' (' + d.n_down + ')', 'papers you marked as not for ' + (all ? 'a tag' : 'this tag'))}
                ${tab('cited', 'Cited (' + d.n_cited + ')', 'papers that the papers you saved' + where + ' cite, the most cited first')}
            </nav>
            ${g.rank === 'tags' ? html`
            <label class="window" title="only papers that came out in this time">
                <select aria-label="time window" value=${g.time_filter}
                        onChange=${e => navigate(withSetting(g, x => (e.target.value ? x.set('time_filter', e.target.value) : x.delete('time_filter'))))}>
                    ${windows.map(([v, label]) => html`<option key=${v} value=${v}>${label}</option>`)}
                </select>
            </label>` : null}
            <${PubFilter} g=${g} />
        </div>
        <p class="muted">
            ${g.rank === 'tags' ? (all && g.mix === 'turns' ? 'Each of your tags picks its best papers in turn; the label on a paper says which.'
                : all ? 'Ranked by how well they match all ' + d.n_saved + ' papers you saved.'
                : 'Ranked by how well they match the ' + d.n_saved + ' papers you saved here.' + (mine ? ' 👍 / 👎 teach it what you want.' : ''))
            : g.rank === 'unread' ? "Saved papers you haven't read yet, newest first." + (mine ? ' Tick "read" on one when you have; it stays until you reload.' : '')
            : g.rank === 'cited' ? 'Papers that the papers you saved' + where + ' cite, the ones most of them cite first; the ones you saved are left out.'
            : g.rank === 'down' ? 'Marked as not for ' + (all ? 'one of your tags' : 'this tag') + ': only that tag learns from them and leaves them out.' + (mine ? ' × undoes one.' : '')
            : 'Newest first.'}
        </p>
    </section>`;
};

// someone else's tag: take it over to vote on it and get its recommendations yourself
const CopyTag = ({ d, g }) => {
    const m = me.use();
    const [as, setAs] = useState(g.tags);
    const [state, run] = useAction();
    if (!m.user || m.readonly) {
        return html`<section class="copytag"><p class="muted"><a href="/profile">Log in</a> to copy this tag to your account and rate its papers yourself.</p></section>`;
    }
    const submit = e => {
        e.preventDefault();
        run(() => api('/copy_tag', { owner: d.owner, tag: g.tags, as }), 'Copying the tag…').then(r => r && navigate(r.url, { own: true }));
    };
    return html`
    <section class="copytag">
        <form onSubmit=${submit}>
            <label>Copy to my tags as
                <input class="form-control" value=${as} onInput=${e => setAs(e.target.value)} maxLength="100" required /></label>
            <button type="submit" class="btn btn-primary btn-small" disabled=${state.busy}>Copy</button>
            <span class="muted">its saved papers and downvotes, into your own tag of that name (merged if you have one)</span>
        </form>
        <${Err} msg=${state.error} />
    </section>`;
};

// how the model ranks; every option says what it does, as a tooltip and in a line below it
const Advanced = ({ g, d }) => {
    const custom = g.svm_c || g.feats !== 'auto' || g.model !== 'auto' || g.mix !== 'one';
    const submit = e => {
        e.preventDefault();
        const q = new URLSearchParams(new FormData(e.target));
        if (d.viewing && d.owner) q.set('u', d.owner);
        navigate('/?' + q.toString());
    };
    const feats = g.has_emb ? ['auto', 'tfidf', 'emb', 'concat', 'fusion'] : ['tfidf'];
    return html`
    <details id="options" open=${custom}>
        <summary>Advanced</summary>
        <form id="optgrid" onSubmit=${submit}>
            ${['rank', 'pid', 'time_filter', 'skip_have'].map(k => html`<input key=${k} type="hidden" name=${k} value=${g[k]} />`)}
            ${g.rank === 'tags' ? html`
            <label title="Which of the tags to rank by: one name, several separated by commas, or all">tags
                <input name="tags" type="text" value=${g.tags} placeholder="e.g. all, or rl,vision" />
                <small>one, several (a,b) or all</small>
            </label>
            <label title="How several tags (or all) are combined. One model: trained on the papers of all the tags together, one ranking (fast). Per tag, in turns: one model per tag, and the tags take turns picking their best paper, each labeled with its tag; better with unrelated topics, but takes a few seconds per tag">several tags
                <select name="mix" value=${g.mix}>
                    <option value="one">one model</option>
                    <option value="turns">per tag, in turns</option>
                </select>
                <small>per tag: better for unrelated topics, slower</small>
            </label>` : html`<input type="hidden" name="tags" value=${g.tags} />`}
            <label title="What a paper looks like to the model. tfidf: the words of its title and abstract. emb: a neural text embedding that also catches meaning without shared words. concat: both together (usually best). fusion: two separate models, scores averaged. auto picks the tuned default">features
                <select name="feats" value=${g.feats}>
                    ${feats.map(f => html`<option key=${f} value=${f}>${f}${f === 'auto' ? ' (' + g.eff_feats + ')' : ''}</option>`)}
                </select>
                <small>how papers are compared; auto is tuned</small>
            </label>
            <label title="The model that learns from your saved papers. svm: every other paper counts as a 'no' (the original arxiv-sanity). logreg: a random sample of papers counts as 'no', and your 'not for this tag' papers more. lda: compares your papers' average with everyone else's, no training loop; much faster, slightly worse matches. auto picks the tuned default">model
                <select name="model" value=${g.model}>
                    ${['auto', 'svm', 'logreg', 'lda'].map(x => html`<option key=${x} value=${x}>${x}${x === 'auto' ? ' (' + g.eff_model + ')' : ''}</option>`)}
                </select>
                <small>what learns from your papers; auto is tuned</small>
            </label>
            <label title="How closely the model follows your saved papers. Lower: smoother, broader matches. Higher: sticks closer to exactly what you saved. Empty = tuned default">regularization C
                <input name="svm_c" type="text" value=${g.svm_c} placeholder=${'default: ' + g.default_c} />
                <small>${g.eff_feats} + ${g.eff_model} uses ${g.default_c}${g.eff_feats === 'fusion' ? ' (tfidf / emb)' : ''} by default; lower = broader matches</small>
            </label>
            <button type="submit" class="btn">Apply</button>
        </form>
    </details>`;
};

// the tools of a ranking: why these papers, how well the tag hangs together, advanced settings
const Tools = ({ d, g, single }) => {
    const [showWords, setShowWords] = useState(false);
    return html`
    <div class="tools">
        ${d.words.length ? html`<button type="button" class="linkbtn" onClick=${() => setShowWords(s => !s)}>
            ${showWords ? 'hide the words' : 'why these papers?'}</button>` : null}
        ${single ? html`<${TagCheck} tag=${g.tags} />` : null}
        <${Advanced} g=${g} d=${d} />
    </div>
    ${showWords && d.words.length ? html`<section id="wordwrap"><${WordList} words=${d.words} desc=${d.words_desc} /></section>` : null}`;
};

const Welcome = ({ d, m }) => {
    if (d.viewing) return null;
    if (!m.user) {
        return html`<div id="log-fun-warn"><b>Arxiv Resanity</b> finds the new arXiv papers you care about: save a few you like into a tag,
            and it ranks every new paper against it, with a weekly digest. <a href="/profile">Log in or sign up</a> to start.</div>`;
    }
    if (m.readonly) return html`<div id="log-fun-warn">👁 Viewing <b>${m.user}</b>'s tags read-only. <a href="/profile">Log in</a> to make changes.</div>`;
    if (d.tags.length) return null;
    return html`
    <section class="card onboarding">
        <h1>Welcome! Three steps to your recommendations</h1>
        <ol>
            <li><b>Find a few papers you like</b>: search below, browse the <a href="/?rank=time">newest</a>, or paste arXiv links on the <a href="/import">add</a> page.</li>
            <li><b>Hit "+ Save to…"</b> on each and give them a tag, like <i>diffusion</i> or <i>robot learning</i>. One tag per topic you follow.</li>
            <li><b>Your tags then show up here</b>, each with the best new papers for it (this page opens on them from then on). "✗ Not for…" on a bad match makes that tag better, and every Sunday your <a href="/digest">digest</a> collects the week's best.</li>
        </ol>
    </section>`;
};

// search on its own: it doesn't take over the filters of the page it was typed on
const Search = ({ q, d }) => {
    const [value, setValue] = useState(q);
    useEffect(() => setValue(q), [q]);
    const submit = e => {
        e.preventDefault();
        const p = new URLSearchParams({ q: e.target.elements.q.value });
        if (d && d.viewing && d.owner) p.set('u', d.owner);
        navigate('/?' + p.toString());
    };
    return html`
    <form id="searchrow" onSubmit=${submit}>
        <input name="q" type="search" id="qfield" value=${value} onInput=${e => setValue(e.target.value)}
               placeholder="Search papers…" aria-label="search" />
        <button type="submit" class="btn btn-primary">Search</button>
    </form>`;
};

// what the page waits for while a model is trained
const waitingFor = url => {
    const q = url.searchParams;
    if (q.get('q')) return 'Searching…';
    if (q.get('rank') === 'pid') return 'Finding similar papers…';
    const tags = q.get('tags');
    if (q.get('rank') === 'tags' && tags && tags !== 'all') return 'Ranking papers for "' + tags + '"…';
    if (q.get('rank') === 'tags' || !q.get('rank')) return 'Finding the best new papers for you…';
    return 'Loading…';
};

export const ListPage = () => {
    const url = location.use();
    const m = me.use();
    const path = listPath(url);
    const { data: d, error, loading, reload, dataPath } = useData(path, waitingFor(url));
    if (!d) {
        if (error) return html`<div class="page"><section class="card"><p class="form-error">${error}</p>
            <button type="button" class="btn" onClick=${reload}>Try again</button></section></div>`;
        return html`<${Search} q=${url.searchParams.get('q') || ''} /><${Skeleton} caption=${waitingFor(url)} />`;
    }
    setShared(d, m);
    const g = d.gvars;
    const userTags = withMadeTags(d.tags.map(t => t.name).filter(n => n !== 'all'));
    const editable = canEdit(m, d);
    const page = { rank: g.rank, tags: g.tags, viewing: d.viewing, owner: d.owner, canEdit: editable, userTags };
    // the next page's data is on its way: a model to train shows a skeleton of the list, else the
    // current list stays (dimmed) until the new one is there
    const want = new URLSearchParams(url.search);
    // (the bare / is "For you", a model, for whoever has tags: likely if logged in or on a shared link)
    const wantRank = want.get('rank') || (m.user || want.get('u') ? 'tags' : 'time');
    const slow = loading && SLOW_RANKS.includes(wantRank) && !want.get('q');
    const single = g.rank === 'tags' && !g.tags.includes(',') && userTags.includes(g.tags);
    const exact = new URLSearchParams(url.search);
    exact.set('exact', 'yes');
    exact.delete('page_number');
    const pageNo = parseInt(g.page_number, 10);
    const goPage = n => {
        const q = new URLSearchParams(url.search);
        q.set('page_number', n);
        navigate('/?' + q.toString());
    };
    return html`
    <${Page.Provider} value=${page}>
        <${ViewBanner} info=${d} />
        <${Welcome} d=${d} m=${m} />
        <${Search} q=${g.search_query} d=${d} />
        ${d.tags.length ? html`<${TagList} tags=${d.tags} g=${g} d=${d} />` : null}
        ${d.tag_view ? html`<${TagHead} d=${d} g=${g} editable=${editable} />` : d.heading ? html`
        <section id="taghead"><h1>${d.heading}</h1><${PubFilter} g=${g} /></section>` : null}
        ${d.tag_view && d.viewing && g.tags !== 'all' && userTags.includes(g.tags) ? html`<${CopyTag} d=${d} g=${g} key=${g.tags} />` : null}
        ${SLOW_RANKS.includes(g.rank) ? html`<${Tools} d=${d} g=${g} single=${single} key=${dataPath} />` : null}
        <${ViewSwitch} share=${Boolean(m.user || d.viewing)} />
        <section id="wrap" class=${loading && !slow ? 'stale' : ''}>
            ${slow ? html`<${Skeleton} caption=${waitingFor(url)} />` : html`
            ${error ? html`<div class="empty"><span class="form-error">${error}</span>
                ${' '}<button type="button" class="linkbtn" onClick=${reload}>Try again</button></div>` : null}
            ${g.corrected ? html`<div class="notice">Showing results for <b>${g.corrected}</b>.
                ${' '}<a href=${'/?' + exact.toString()}>Search for “${g.search_query}” instead</a></div>` : null}
            ${d.notice ? html`<div class="notice">${d.notice}</div>` : null}
            <${PaperList} papers=${d.papers} key=${dataPath} />`}
        </section>
        <nav id="pagination" aria-label="pages">
            <button class="btn" disabled=${pageNo <= 1} onClick=${() => goPage(pageNo - 1)}>← prev</button>
            <span>page ${pageNo}</span>
            <button class="btn" disabled=${!d.has_next} onClick=${() => goPage(pageNo + 1)}>next →</button>
        </nav>
    </${Page.Provider}>`;
};
