// the Markdown of users' notes, as Preact nodes: never html from the text (no innerHTML, so a note
// can't inject anything), links only to http(s), mailto and this site. what notes need: paragraphs
// and line breaks, # headings, - and 1. lists, > quotes, ``` code blocks, **bold**, *italic*,
// ~~struck~~, `code`, [links](url), bare urls, and $math$ (typeset by KaTeX like the abstracts)
import { html } from 'app/lib';
import { MathText } from 'app/math';

const MATH = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|(?<!\\)\$(?:[^$\\\n]|\\[\s\S])+?\$)/;
const INLINE = new RegExp([
    /(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/.source, // 1, 2: code
    /\[([^\]\n]+)\]\(([^)\s]+)\)/.source, // 3, 4: link
    /(https?:\/\/[^\s<>()]*[^\s<>().,;:!?'"])/.source, // 5: bare url
    /\*\*(?!\s)([\s\S]+?)\*\*|__(?!\s)([\s\S]+?)__/.source, // 6, 7: bold
    /~~(?!\s)([\s\S]+?)~~/.source, // 8: struck
    /\*(?![\s*])([^*]+?)\*|(?<!\w)_(?![\s_])([^_]+?)_(?!\w)/.source, // 9, 10: italic
].join('|'));

const safeUrl = u => (/^(https?:\/\/|mailto:|\/(?!\/))/i.test(u) ? u : null);

// text without markup: math stays math, the rest goes to KaTeX's neighbour MathText only if it has some
const text = (s, key) => (s.includes('$') || s.includes('\\') ? html`<${MathText} key=${key} text=${s} />` : s);

function inline(s, depth = 0) {
    const out = [];
    // math first, so the underscores and stars of $a_i * b_j$ stay math
    s.split(MATH).forEach((part, i) => {
        if (i % 2) { out.push(html`<${MathText} key=${'m' + out.length} text=${part} />`); return; }
        let rest = part;
        while (rest) {
            const m = depth < 4 ? INLINE.exec(rest) : null;
            if (!m) { out.push(text(rest, 't' + out.length)); break; }
            if (m.index) out.push(text(rest.slice(0, m.index), 't' + out.length));
            const k = 'i' + out.length;
            if (m[2] !== undefined) out.push(html`<code key=${k}>${m[2]}</code>`);
            else if (m[3] !== undefined) {
                const u = safeUrl(m[4]);
                out.push(u ? html`<a key=${k} href=${u} rel="noopener noreferrer">${inline(m[3], depth + 1)}</a>` : m[0]);
            } else if (m[5] !== undefined) out.push(html`<a key=${k} href=${m[5]} rel="noopener noreferrer">${m[5]}</a>`);
            else if (m[6] !== undefined || m[7] !== undefined) out.push(html`<strong key=${k}>${inline(m[6] ?? m[7], depth + 1)}</strong>`);
            else if (m[8] !== undefined) out.push(html`<s key=${k}>${inline(m[8], depth + 1)}</s>`);
            else out.push(html`<em key=${k}>${inline(m[9] ?? m[10], depth + 1)}</em>`);
            rest = rest.slice(m.index + m[0].length);
        }
    });
    return out;
}

// lines kept as they were typed: a line break in a paragraph is one on screen
const lines = (ls) => ls.flatMap((l, i) => (i ? [html`<br key=${'b' + i} />`, ...inline(l)] : inline(l)));

const LIST = /^\s*([-*+]|\d+[.)])\s+(.*)$/;

export function markdown(src) {
    const ls = String(src || '').replace(/\r\n?/g, '\n').split('\n');
    const blocks = [];
    let i = 0;
    while (i < ls.length) {
        const l = ls[i];
        const key = 'k' + blocks.length;
        if (!l.trim()) { i++; continue; }
        if (/^\s*```/.test(l)) {
            const body = [];
            for (i++; i < ls.length && !/^\s*```/.test(ls[i]); i++) body.push(ls[i]);
            i++;
            blocks.push(html`<pre key=${key}><code>${body.join('\n')}</code></pre>`);
            continue;
        }
        const h = l.match(/^\s*(#{1,6})\s+(.*)$/);
        if (h) {
            blocks.push(html`<p key=${key} class="md-h">${inline(h[2])}</p>`);
            i++;
            continue;
        }
        if (/^\s*>/.test(l)) {
            const body = [];
            for (; i < ls.length && /^\s*>/.test(ls[i]); i++) body.push(ls[i].replace(/^\s*>\s?/, ''));
            blocks.push(html`<blockquote key=${key}>${lines(body)}</blockquote>`);
            continue;
        }
        const li = l.match(LIST);
        if (li) {
            const ordered = /\d/.test(li[1]);
            const items = [];
            for (; i < ls.length; i++) {
                const m = ls[i].match(LIST);
                if (m && /\d/.test(m[1]) === ordered) items.push([m[2]]);
                else if (ls[i].trim() && /^\s+/.test(ls[i]) && items.length) items[items.length - 1].push(ls[i].trim());
                else break;
            }
            const lis = items.map((it, j) => html`<li key=${j}>${lines(it)}</li>`);
            blocks.push(ordered ? html`<ol key=${key}>${lis}</ol>` : html`<ul key=${key}>${lis}</ul>`);
            continue;
        }
        const body = [];
        for (; i < ls.length && ls[i].trim() && !LIST.test(ls[i]) && !/^\s*(```|#{1,6}\s|>)/.test(ls[i]); i++) body.push(ls[i]);
        blocks.push(html`<p key=${key}>${lines(body)}</p>`);
    }
    return blocks;
}

// one line of a note without its markup, for the compact cards and tooltips
export const plainText = src => String(src || '')
    .replace(/```[\s\S]*?(```|$)/g, ' ')
    .replace(/\[([^\]\n]+)\]\([^)\s]+\)/g, '$1')
    .replace(/^\s*(#{1,6}|>|[-*+]|\d+[.)])\s+/gm, '')
    .replace(/(\*\*|__|~~|`)/g, '')
    .replace(/\s+/g, ' ')
    .trim();
