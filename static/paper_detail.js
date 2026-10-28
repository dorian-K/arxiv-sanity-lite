'use strict';

const PaperLite = ({ paper: p }) => html`
    <article class="rel_paper">
        <h2 class="rel_title"><a href=${'https://arxiv.org/abs/' + p.id}><${MathText} text=${p.title} /></a></h2>
        <div class="rel_authors">${p.authors}</div>
        <div class="rel_meta">
            <span class="rel_time">${p.time}</span>
            <span class="rel_tags">${p.tags}</span>
        </div>
        <div class="rel_abs"><${MathText} text=${p.summary} /></div>
        <div class="rel_links">
            <a href=${'https://arxiv.org/pdf/' + p.id}>pdf</a>
            <a href=${'/?rank=pid&pid=' + p.id}>similar</a>
        </div>
    </article>`;

mount(html`<${PaperLite} paper=${paper} />`, 'wrap');
