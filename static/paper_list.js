'use strict';

// the user's tag names, for autocompletion in the tag editor
// logged in with the password (not a read-only view of a shared account)
const canEdit = Boolean(user) && !readonly;
const userTagNames = (typeof tags !== 'undefined' ? tags : []).map(t => t.name).filter(n => n !== 'all');

// calls one of the tag endpoints; resolves on "ok...", otherwise shows the server's message
function tagRequest(path) {
    return fetch(path, { method: 'POST' }).then(r => r.text()).then(t => {
        if (!t.startsWith('ok')) {
            alert(t);
            throw new Error(t);
        }
        return t;
    });
}

// a tag the paper is in: the name links to the tag's page, x takes the paper out of it
const UTag = ({ tag, onRemove }) => html`
    <span class="chip chip-tag chip-removable">
        <a href=${'/?rank=tags&tags=' + encodeURIComponent(tag)}>${tag}</a>
        <button class="chip-x" title=${'remove from ' + tag} aria-label=${'remove from ' + tag} onClick=${onRemove}>×</button>
    </span>`;

// a tag the paper was marked "not for": x undoes that
const NTag = ({ tag, onUndo }) => html`
    <span class="chip chip-neg chip-removable" title=${'marked as not for ' + tag}>
        <span>not for ${tag}</span>
        <button class="chip-x" title="undo" aria-label=${'undo not for ' + tag} onClick=${onUndo}>×</button>
    </span>`;

// picks the tag for "save" / "not for me": the user's tags as one-tap chips, or a new name
const TagPicker = ({ mode, tags, onPick, onCancel }) => {
    const [value, setValue] = useState('');
    const submit = e => {
        e.preventDefault();
        const tag = value.trim();
        if (tag) onPick(tag);
    };
    const save = mode === 'add';
    return html`
        <div class=${'tag-picker tag-picker-' + mode}>
            <div class="tag-picker-label">${save ? 'Save to' : 'Not what this tag is about:'}</div>
            ${tags.length ? html`
            <div class="tag-picker-tags">
                ${tags.map(t => html`<button key=${t} class="chip chip-pick" onClick=${() => onPick(t)}>${t}</button>`)}
            </div>` : null}
            <form class="tag-picker-new" onSubmit=${submit}>
                <input autoFocus=${tags.length === 0} value=${value} onInput=${e => setValue(e.target.value)}
                       onKeyDown=${e => e.key === 'Escape' && onCancel()} maxLength="100"
                       placeholder=${tags.length ? 'or a new tag…' : 'name a tag, e.g. diffusion'} aria-label="new tag" />
                <button type="submit" class="btn btn-small btn-primary" disabled=${!value.trim()}>${save ? 'Save' : 'OK'}</button>
                <button type="button" class="btn btn-small" onClick=${onCancel}>Cancel</button>
            </form>
        </div>`;
};

// rankings whose scores mean something to show next to the title
const SHOW_SCORE = ['tags', 'pid', 'search'];

// a rank fraction as people read it: 0.0031 -> "top 0.3%"
const fmtPct = x => 'top ' + (x < 0.001 ? '0.1' : x < 0.1 ? (100 * x).toFixed(1) : Math.round(100 * x)) + '%';

// held-out ranks of the current tag's papers, filled in by a tag check (see TagCheck)
const heldout = { tag: null, ranks: {}, misfit: 0.1 };
const useHeldout = () => {
    const [, bump] = useState(0);
    useEffect(() => {
        const f = () => bump(n => n + 1);
        window.addEventListener('heldout', f);
        return () => window.removeEventListener('heldout', f);
    }, []);
    return heldout;
};

// one-click votes for a given tag (the digest page): like, neutral (seen, no opinion: hidden
// from later digests, not used for training) or downvote. one state per paper and tag,
// clicking the active one undoes it
const VOTES = [
    { key: 'like', label: '👍 Like', on: '👍 Liked', add: '/add/', sub: '/sub/' },
    { key: 'seen', label: '😐 Neutral', on: '😐 Neutral', add: '/seen/', sub: '/unseen/' },
    { key: 'down', label: '👎 Not for me', on: '👎 Not for me', add: '/addneg/', sub: '/subneg/' },
];
const QuickVote = ({ pid, tag, state, onChange }) => {
    const enc = pid + '/' + encodeURIComponent(tag);
    const click = v => tagRequest((state === v.key ? v.sub : v.add) + enc)
        .then(() => onChange(state === v.key ? null : v.key)).catch(() => {});
    return html`
        <div class="quickvote">
            ${VOTES.map(v => html`
            <button key=${v.key} class=${'btn btn-small vote-' + v.key + (state === v.key ? ' on' : '')}
                    aria-pressed=${state === v.key} onClick=${() => click(v)}>${state === v.key ? v.on : v.label}</button>`)}
        </div>`;
};

const Paper = ({ paper: p, quickTag }) => {
    const [utags, setUtags] = useState(p.utags);
    const [ntags, setNtags] = useState(p.ntags);
    const [stags, setStags] = useState(p.stags || []);
    const [picker, setPicker] = useState(null); // null | 'add' | 'neg'
    const [open, setOpen] = useState(false);

    const without = (list, tag) => list.filter(t => t !== tag);
    const withTag = (list, tag) => (list.includes(tag) ? list : [...list, tag]);

    const enc = tag => p.id + '/' + encodeURIComponent(tag);
    // mirror the server: saving and "not for" one tag exclude each other
    const save = tag => tagRequest('/add/' + enc(tag)).then(() => {
        setUtags(u => withTag(u, tag)); setNtags(n => without(n, tag)); setPicker(null);
    }).catch(() => {});
    const notFor = tag => tagRequest('/addneg/' + enc(tag)).then(() => {
        setNtags(n => withTag(n, tag)); setUtags(u => without(u, tag)); setPicker(null);
    }).catch(() => {});
    const unsave = tag => tagRequest('/sub/' + enc(tag)).then(() => setUtags(u => without(u, tag))).catch(() => {});
    const undoNeg = tag => tagRequest('/subneg/' + enc(tag)).then(() => setNtags(n => without(n, tag))).catch(() => {});
    const toggle = mode => setPicker(m => (m === mode ? null : mode));
    // offer the tags this paper isn't in yet; new tags typed here join the list for the next paper
    const pickable = (picker === 'add' ? userTagNames.filter(t => !utags.includes(t)) : userTagNames.filter(t => !ntags.includes(t)));
    const ho = useHeldout();
    const fit = ho.tag && utags.includes(ho.tag) ? ho.ranks[p.id] : undefined;
    const long = p.summary.length > 360;
    const showScore = SHOW_SCORE.includes(gvars.rank);

    return html`
    <article class="rel_paper">
        ${p.thumb_url ? html`<div class="rel_img"><img src=${p.thumb_url} alt="" loading="lazy" /></div>` : null}
        <div class="rel_head">
            ${p.pct !== undefined
                ? html`<span class="rel_score" title=${'rank among all papers (model score ' + p.weight.toFixed(2) + ')'}>${fmtPct(p.pct)}</span>`
                : showScore ? html`<span class="rel_score" title="search score">${p.weight.toFixed(2)}</span>` : null}
            <h2 class="rel_title"><a href=${'https://arxiv.org/abs/' + p.id}><${MathText} text=${p.title} /></a></h2>
        </div>
        ${fit !== undefined ? html`
        <div class=${'rel_fit' + (fit > ho.misfit ? ' misfit' : '')}
             title="where this paper lands when the tag's other papers rank it (it isn't used to train that model)">
            ${fit > ho.misfit ? "doesn't fit " + ho.tag + ' well: ' : 'fits ' + ho.tag + ': '}${fmtPct(fit)} when held out
        </div>` : null}
        <div class="rel_authors">${p.authors}</div>
        <div class="rel_meta">
            <span class="rel_time">${p.time}</span>
            <span class="rel_tags">${p.tags}</span>
        </div>
        ${canEdit && quickTag ? html`
        <${QuickVote} pid=${p.id} tag=${quickTag}
            state=${utags.includes(quickTag) ? 'like' : ntags.includes(quickTag) ? 'down' : stags.includes(quickTag) ? 'seen' : null}
            onChange=${v => {
                // the server keeps one state per paper and tag, so the others are gone now
                setUtags(u => (v === 'like' ? withTag(u, quickTag) : without(u, quickTag)));
                setNtags(n => (v === 'down' ? withTag(n, quickTag) : without(n, quickTag)));
                setStags(x => (v === 'seen' ? withTag(x, quickTag) : without(x, quickTag)));
            }} />` : null}
        ${canEdit && !quickTag ? html`
        <div class="rel_utags">
            <button class=${'btn btn-small rel_utag_add' + (picker === 'add' ? ' on' : '')} onClick=${() => toggle('add')}>+ Save</button>
            <button class=${'btn btn-small rel_utag_neg' + (picker === 'neg' ? ' on' : '')} onClick=${() => toggle('neg')}
                    title="not what one of your tags is about: improves its recommendations">✗ Not for me</button>
            ${utags.map(t => html`<${UTag} key=${'u' + t} tag=${t} onRemove=${() => unsave(t)} />`)}
            ${ntags.map(t => html`<${NTag} key=${'n' + t} tag=${t} onUndo=${() => undoNeg(t)} />`)}
        </div>` : null}
        ${picker ? html`<${TagPicker} mode=${picker} tags=${pickable}
            onPick=${t => { if (!userTagNames.includes(t)) userTagNames.push(t); (picker === 'add' ? save : notFor)(t); }}
            onCancel=${() => setPicker(null)} />` : null}
        <div class=${'rel_abs' + (long && !open ? ' clamped' : '')} onClick=${() => long && !open && setOpen(true)}>
            <${MathText} text=${p.summary} />
        </div>
        <div class="rel_links">
            ${long ? html`<button class="linkbtn" onClick=${() => setOpen(o => !o)}>${open ? 'show less' : 'show more'}</button>` : null}
            <a href=${'https://arxiv.org/pdf/' + p.id}>pdf</a>
            <a class="rel_more" href=${'/?rank=pid&pid=' + p.id}>similar</a>
            <a class="rel_inspect" href=${'/inspect?pid=' + p.id}>inspect</a>
        </div>
    </article>`;
};

const PaperList = ({ papers }) => {
    if (papers.length === 0) {
        return html`<div class="empty">No papers here. Try another search, a longer time window, or tag a few papers first.</div>`;
    }
    return html`<div id="paperList" class="rel_papers">${papers.map(p => html`<${Paper} key=${p.id} paper=${p} />`)}</div>`;
};

const Tag = ({ tag: t }) => {
    const del = e => {
        e.preventDefault();
        if (!confirm("delete the tag '" + t.name + "' and its downvotes? the papers stay in the database")) return;
        tagRequest('/del/' + encodeURIComponent(t.name)).then(() => window.location.reload()).catch(() => {});
    };
    const active = gvars.rank === 'tags' && gvars.tags.split(',').includes(t.name);
    return html`
        <span class=${'chip chip-tag chip-count' + (t.name === 'all' ? ' rel_utag_all' : '') + (active ? ' active' : '')}>
            <a href=${'/?rank=tags&tags=' + encodeURIComponent(t.name)}>${t.name}${t.n !== undefined ? html` <small>${t.n}</small>` : null}</a>
            ${t.name !== 'all' && canEdit ? html`<button class="chip-x" title=${'delete tag ' + t.name} aria-label=${'delete tag ' + t.name} onClick=${del}>×</button>` : null}
        </span>`;
};

const TagList = ({ tags }) => {
    const [shown, setShown] = useState(false);
    const toggleWords = () => {
        const w = document.getElementById('wordwrap');
        w.hidden = !w.hidden;
        setShown(!w.hidden);
    };
    return html`
        <div id="tagList">
            <span class="taglabel">Your tags</span>
            ${tags.map(t => html`<${Tag} key=${t.name} tag=${t} />`)}
            ${words.length > 0 ? html`<button id="inspect_svm" class="linkbtn" onClick=${toggleWords}>${shown ? 'hide' : 'why these papers?'}</button>` : null}
        </div>`;
};

// the digest page: one section per tag, with quick votes for that tag
const Digest = ({ sections }) => html`
    ${sections.map(s => html`
    <section key=${s.tag} class="digest-section">
        <h2 class="digest-tag"><a href=${'/?rank=tags&tags=' + encodeURIComponent(s.tag)}>${s.tag}</a></h2>
        <div class="rel_papers">${s.papers.map(p => html`<${Paper} key=${p.id} paper=${p} quickTag=${s.tag} />`)}</div>
    </section>`)}`;

// "check this tag": how well the tag's papers predict each other, and which don't fit
const TagCheck = ({ tag }) => {
    const [state, setState] = useState(null); // null | 'loading' | result | {error}
    const run = () => {
        setState('loading');
        fetch('/tag_check?tag=' + encodeURIComponent(tag)).then(r => r.json()).then(res => {
            setState(res);
            if (res.ranks) {
                Object.assign(heldout, { tag, ranks: res.ranks, misfit: res.misfit_threshold });
                window.dispatchEvent(new Event('heldout'));
            }
        }).catch(() => setState({ error: 'the check failed, try again' }));
    };
    if (state === null || state === 'loading') {
        return html`<div class="tagcheck">
            <button class="btn btn-small" onClick=${run} disabled=${state === 'loading'}>
                ${state === 'loading' ? 'Checking…' : 'Check how well "' + tag + '" hangs together'}</button></div>`;
    }
    if (state.error) return html`<div class="tagcheck"><span class="form-error">${state.error}</span></div>`;
    if (!state.median) {
        return html`<div class="tagcheck muted">Save at least ${state.min_papers} papers to "${tag}" to check it.</div>`;
    }
    const m = state.misfits;
    return html`
        <div class="tagcheck card">
            <p>
                Each of your ${state.n} papers in <b>${tag}</b> was ranked by a model trained on the others.
                A typical one lands in the <b>${fmtPct(state.median)}</b> of all papers, 90% of them within
                the ${fmtPct(state.p90)}. ${m.length === 0 ? 'They all fit together well.' : ''}
            </p>
            ${m.length ? html`
            <p>${m.length === 1 ? "This paper doesn't" : 'These ' + m.length + " papers don't"} fit with the rest
                (outside the ${fmtPct(state.misfit_threshold)}):</p>
            <ul class="misfits">${m.map(x => html`<li key=${x.id}>
                <a href=${'/?rank=pid&pid=' + x.id}><${MathText} text=${x.title} /></a>
                <span class="muted"> · ${fmtPct(x.rank)}</span></li>`)}</ul>
            <p class="muted">
                ${m.length >= 3
                    ? "If they're about another topic, give them a tag of their own: a side topic in a shared tag gets crowded out of the top of the list, and on its own it gets its own list."
                    : m.length === 1 ? 'Maybe it belongs in another tag, or it is a one-off.' : 'Maybe they belong in another tag, or are one-offs.'}
                ${' '}Remove a paper from a tag with the × on its tag.
            </p>` : null}
        </div>`;
};

if (typeof digest !== 'undefined') {
    mount(html`<${Digest} sections=${digest} />`, 'digestwrap');
} else {
    mount(html`<${PaperList} papers=${papers} />`, 'wrap');
    mount(html`<${TagList} tags=${tags} />`, 'tagwrap');
    const single = gvars.rank === 'tags' && !gvars.tags.includes(',') && userTagNames.includes(gvars.tags);
    if (single) mount(html`<${TagCheck} tag=${gvars.tags} />`, 'tagcheckwrap');
}
