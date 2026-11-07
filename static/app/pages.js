// the digest, one paper's words, the stats and the text pages (about, privacy, imprint)
import { html, setShared, me, api, useData, location, canEdit, Err, useAction, LoadError } from 'app/lib';
import { Page, ViewSwitch, DigestSections, WordList, ViewBanner, withMadeTags } from 'app/papers';
import { MathText } from 'app/math';

// -----------------------------------------------------------------------------
// the digest: the user's latest one (what the email showed), with quick votes per tag

export const DigestPage = () => {
    const url = location.use();
    const m = me.use();
    const { data: d, error, reload, dataPath } = useData('/ui/digest' + url.search, 'Loading your digest…');
    const [state, run] = useAction();
    if (!d) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    setShared(d, m);
    const editable = canEdit(m, d);
    const page = { rank: 'digest', tags: '', viewing: d.viewing, owner: d.owner, canEdit: editable, userTags: withMadeTags(d.tags.map(t => t.name)) };
    const make = () => run(() => api('/digest', {}), 'Making your digest… one model per tag, this takes a few seconds per tag.').then(r => r && reload());
    return html`
    <${Page.Provider} value=${page}>
        <${ViewBanner} info=${d} />
        <div class="page">
            <section class="card">
                <h1>${d.not_shared ? 'Digest' : d.viewing ? '@' + d.owner + "'s digest" : 'Your digest'}</h1>
                ${d.not_shared ? html`<p class="muted">Nothing to show here.</p>`
                : !d.owner ? html`<p class="muted">Please <a href="/profile">log in</a> to see your digest.</p>` : html`
                <${Err} msg=${state.error} />
                ${d.sections.length ? html`<p class="muted">
                    The best papers of the ${d.digest_days} days before ${d.digest_time} for each of your tags.
                    👍 adds a paper to the tag and 👎 marks it as not what the tag is about; both shape next week's list. 😐 just means "seen": the paper won't come back, but the tag learns nothing from it.
                    Click a tag's name to collapse it.
                </p>` : d.digest_time ? html`<p class="muted">Nothing new for your tags in the ${d.digest_days} days before ${d.digest_time}.</p>`
                : html`<p class="muted">
                    No digest yet. It's made every week when the emails go out
                    (add your email on your <a href="/profile">profile</a>), or right now:
                </p>`}
                ${editable ? html`<button type="button" class="btn" disabled=${state.busy} onClick=${make}>
                    ${d.digest_time ? 'Make a fresh one' : 'Make my digest'}</button>` : null}`}
            </section>
            ${d.sections.length ? html`
            <${ViewSwitch} share=${true} />
            <${DigestSections} sections=${d.sections} key=${dataPath + d.digest_time} />` : null}
        </div>
    </${Page.Provider}>`;
};

// -----------------------------------------------------------------------------
// one paper and the words it is indexed by

export const InspectPage = () => {
    const url = location.use();
    const { data: d, error, reload } = useData('/ui/inspect' + url.search);
    if (!d) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    const p = d.paper;
    return html`
    <div id="wrap">
        <article class="rel_paper">
            <h2 class="rel_title"><a href=${p.url}><${MathText} text=${p.title} /></a></h2>
            <div class="rel_authors">${p.authors}</div>
            <div class="rel_meta">
                <span class="rel_time">${p.time}</span>
                <span class="rel_tags">${p.tags}</span>
            </div>
            <div class="rel_abs"><${MathText} text=${p.summary} /></div>
            <div class="rel_links">
                <a href=${p.pdf}>pdf</a>
                <a href=${'/?rank=pid&pid=' + p.id}>similar</a>
            </div>
        </article>
    </div>
    <div id="wordwrap"><${WordList} words=${d.words} desc=${d.words_desc} /></div>`;
};

// -----------------------------------------------------------------------------
// what the server has and how fast it is

const Table = ({ head, rows }) => html`
    <div class="tablescroll"><table class="stattable perftable">
        <tr>${head.map(x => html`<th key=${x}>${x}</th>`)}</tr>
        ${rows.map((r, i) => html`<tr key=${i}>${r.map((x, j) => html`<td key=${j}>${x}</td>`)}</tr>`)}
    </table></div>`;

export const StatsPage = () => {
    const { data: d, error, reload } = useData('/ui/stats');
    if (!d) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    const e = d.embed, r = d.enrich;
    return html`
    <div id="statswrap" class="page">
        <section class="card">
            <h1>Current index</h1>
            <div class="statgrid">
                <div><span class="statnum">${d.num_papers}</span><span class="muted">papers in total</span></div>
                <div><span class="statnum">${d.earliest_paper}</span><span class="muted">earliest paper</span></div>
                <div><span class="statnum">${d.latest_paper}</span><span class="muted">latest paper</span></div>
            </div>
        </section>
        <section class="card">
            <h1>New papers</h1>
            <table class="stattable">
                ${d.new_papers.map(([h, n]) => html`<tr key=${h}><td>last ${h} hours</td><td>${n}</td></tr>`)}
            </table>
        </section>
        <section class="card">
            <h1>Page speed</h1>
            <p class="muted">The last 24 hours, measured on the server (the network adds to it). Half of
                the requests were faster than the median, 95% faster than "95%".</p>
            ${d.pages.length ? html`<${Table} head=${['Page', 'Requests', 'Median', '95%', 'Slowest']}
                rows=${d.pages.map(p => [p.label, p.n, p.p50, p.p95, p.max])} />` : html`<p>Nothing measured yet.</p>`}
        </section>
        <section class="card">
            <h1>Background work</h1>
            <h2>Embedding papers</h2>
            ${e ? html`<p>Last 24 hours: ${e.papers} papers in ${e.sessions} sessions, ${e.busy} of work
                ${e.per_paper ? ' (' + e.per_paper + ' per paper)' : ''}; last session ${e.last}.
                ${' '}${e.left ? e.left + ' papers still without an embedding' + (e.eta ? ', about ' + e.eta + ' at this pace' : '') + '.'
                : e.left === 0 ? 'Every paper has an embedding.' : ''}</p>`
            : html`<p class="muted">No embedding session in the last week.</p>`}
            <h2>Rebuilding the index</h2>
            ${d.computes.length ? html`<${Table} head=${['Finished', 'Took', 'Papers', 'With embedding', 'Peak memory', 'Steps']}
                rows=${d.computes.map(c => [c.ago, c.took, c.papers, c.embedded, c.peak + ' MB', html`<span class="muted">${c.phases}</span>`])} />`
            : html`<p class="muted">No rebuild in the last week.</p>`}
            <h2>References and thumbnails</h2>
            ${r && (r.done || r.error) ? html`<p>${r.done} papers done${r.error ? ', ' + r.error + ' failed (tried again later)' : ''}.
                References read from ${r.html} HTML versions and ${r.pdf} PDFs: ${r.refs} references,
                ${' '}${r.matched} of them to a paper on arXiv. Thumbnails: ${r.figure} figures, ${r.page} first pages.
                Last 24 hours: ${r.day} papers${r.left ? ', ' + r.left + ' waiting' : ''}.
                ${' '}${r.titles ? 'References are matched against ' + r.titles + ' arXiv titles.' : 'No table of all arXiv titles yet (import_snapshot.py --titles).'}</p>`
            : html`<p class="muted">Nothing yet (enrich_daemon.py, ENRICH_ENABLED=1).</p>`}
            <h2>Fetching new papers from arXiv</h2>
            ${d.fetches.length ? html`<${Table} head=${['Finished', 'Took', 'New or updated']}
                rows=${d.fetches.map(f => [f.ago, f.took, f.new + (f.ok ? '' : " (arXiv didn't answer)")])} />`
            : html`<p class="muted">No fetch in the last week.</p>`}
        </section>
    </div>`;
};

// -----------------------------------------------------------------------------
// about, privacy, imprint: html made on the server from templates/pages/

export const TextPage = ({ name }) => {
    const { data: d, error, reload } = useData('/ui/page/' + name);
    if (!d) return error ? html`<${LoadError} error=${error} reload=${reload} />` : null;
    return html`<div dangerouslySetInnerHTML=${{ __html: d.html }} />`;
};
