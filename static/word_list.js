'use strict';

// the most positive / negative tfidf words of the ranking model
const Word = ({ word: w }) => html`
    <span class=${'rel_word ' + (w.weight >= 0 ? 'pos' : 'neg')} title=${'weight ' + w.weight.toFixed(3)}>
        <span class="rel_word_txt">${w.word}</span>
        <span class="rel_word_weight">${w.weight.toFixed(2)}</span>
    </span>`;

const WordList = ({ words, words_desc }) => html`
    <div class="card">
        <p class="muted">${words_desc}</p>
        <div id="wordList" class="rel_words">${words.map((w, ix) => html`<${Word} key=${ix} word=${w} />`)}</div>
    </div>`;

// shown with the page, or once results computed in the background arrive (see Results)
const showWords = w => w.length && mount(html`<${WordList} words=${w} words_desc=${words_desc} />`, 'wordwrap');
if (words.length) showWords(words);
else window.addEventListener('words', e => showWords(e.detail), { once: true });
