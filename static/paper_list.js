'use strict';

// the user's tag names, for autocompletion in the tag editor
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

const UTag = ({ tag }) => html`
    <a class="chip chip-tag" href=${'/?rank=tags&tags=' + encodeURIComponent(tag)}>${tag}</a>`;

const NTag = ({ tag, onUndo }) => html`
    <button class="chip chip-neg" title="downvoted for this tag, click to undo" onClick=${onUndo}>${tag}</button>`;

const EDITOR_LABELS = {
    add: 'add to tag',
    sub: 'remove from tag',
    neg: 'not for tag',
};

// inline replacement for prompt(): a text field with the user's tags as suggestions
const TagEditor = ({ mode, suggestions, onSubmit, onCancel }) => {
    const [value, setValue] = useState('');
    const listId = 'tagsugg-' + mode;
    const submit = e => {
        e.preventDefault();
        const tag = value.trim();
        if (tag) onSubmit(tag);
    };
    return html`
        <form class=${'tag-editor tag-editor-' + mode} onSubmit=${submit}>
            <input autoFocus list=${listId} value=${value} placeholder=${EDITOR_LABELS[mode] + '…'}
                   aria-label=${EDITOR_LABELS[mode]}
                   onInput=${e => setValue(e.target.value)}
                   onKeyDown=${e => e.key === 'Escape' && onCancel()} />
            <datalist id=${listId}>${suggestions.map(s => html`<option key=${s} value=${s} />`)}</datalist>
            <button type="submit" class="btn btn-small btn-primary">${mode === 'sub' ? 'remove' : 'save'}</button>
            <button type="button" class="btn btn-small" onClick=${onCancel}>cancel</button>
        </form>`;
};

const SHOW_SCORE = ['tags', 'pid', 'search'];

const Paper = ({ paper: p }) => {
    const [utags, setUtags] = useState(p.utags);
    const [ntags, setNtags] = useState(p.ntags);
    const [editor, setEditor] = useState(null); // null | 'add' | 'sub' | 'neg'
    const [open, setOpen] = useState(false);

    const without = (list, tag) => list.filter(t => t !== tag);
    const withTag = (list, tag) => (list.includes(tag) ? list : [...list, tag]);

    const apply = tag => {
        const enc = p.id + '/' + encodeURIComponent(tag);
        const req = { add: '/add/', sub: '/sub/', neg: '/addneg/' }[editor] + enc;
        const mode = editor;
        tagRequest(req).then(() => {
            // mirror the server: liking and downvoting for one tag exclude each other
            if (mode === 'add') { setUtags(u => withTag(u, tag)); setNtags(n => without(n, tag)); }
            if (mode === 'sub') { setUtags(u => without(u, tag)); }
            if (mode === 'neg') { setNtags(n => withTag(n, tag)); setUtags(u => without(u, tag)); }
            setEditor(null);
        }).catch(() => {});
    };
    const undoNeg = tag => {
        if (!confirm("remove downvote of this paper for tag '" + tag + "'?")) return;
        tagRequest('/subneg/' + p.id + '/' + encodeURIComponent(tag)).then(() => setNtags(n => without(n, tag))).catch(() => {});
    };

    const toggle = mode => setEditor(e => (e === mode ? null : mode));
    const suggestions = editor === 'sub' ? utags : userTagNames;
    const long = p.summary.length > 360;
    const showScore = SHOW_SCORE.includes(gvars.rank);

    return html`
    <article class="rel_paper">
        ${p.thumb_url ? html`<div class="rel_img"><img src=${p.thumb_url} alt="" loading="lazy" /></div>` : null}
        <div class="rel_head">
            ${showScore ? html`<span class="rel_score" title="ranking score">${p.weight.toFixed(2)}</span>` : null}
            <h2 class="rel_title"><a href=${'https://arxiv.org/abs/' + p.id}><${MathText} text=${p.title} /></a></h2>
        </div>
        <div class="rel_authors">${p.authors}</div>
        <div class="rel_meta">
            <span class="rel_time">${p.time}</span>
            <span class="rel_tags">${p.tags}</span>
        </div>
        ${user ? html`
        <div class="rel_utags">
            <button class="iconbtn rel_utag_add" title="add to a tag" aria-label="add to a tag" onClick=${() => toggle('add')}>+</button>
            <button class="iconbtn rel_utag_sub" title="remove from a tag" aria-label="remove from a tag" onClick=${() => toggle('sub')} disabled=${utags.length === 0}>−</button>
            <button class="iconbtn rel_utag_neg" title="downvote: not what a tag is about" aria-label="downvote for a tag" onClick=${() => toggle('neg')}>✗</button>
            ${utags.map(t => html`<${UTag} key=${'u' + t} tag=${t} />`)}
            ${ntags.map(t => html`<${NTag} key=${'n' + t} tag=${t} onUndo=${() => undoNeg(t)} />`)}
        </div>` : null}
        ${editor ? html`<${TagEditor} mode=${editor} suggestions=${suggestions} onSubmit=${apply} onCancel=${() => setEditor(null)} />` : null}
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
            ${t.name !== 'all' ? html`<button class="chip-x" title=${'delete tag ' + t.name} aria-label=${'delete tag ' + t.name} onClick=${del}>×</button>` : null}
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

mount(html`<${PaperList} papers=${papers} />`, 'wrap');
mount(html`<${TagList} tags=${tags} />`, 'tagwrap');
