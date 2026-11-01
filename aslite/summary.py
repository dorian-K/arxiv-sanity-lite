"""
Short AI summaries of papers (problem, approach, result, caveat), shown on paper cards above the
abstract. enrich_daemon.py makes them from the paper's full text, which it downloads anyway for
the references (arXiv's HTML version, else the PDF), so they cost no extra request to arxiv.org
for new papers. Off unless SUMMARIES_ENABLED=1, with OPENROUTER_API_KEY set.

The model goes through OpenRouter. What it gets is the paper's public text, never anything about
a user. Chosen on 11 speech papers against 15 other models (GPT-6 Luna, GLM 5.3 Flash, DeepSeek,
Gemma, ...): MiMo V2.6 Flash made no factual error that was found, gave the most specific
caveats, and costs about $0.001 per paper. The prompt is in the system message and once more
after the paper (that kept the models shorter), and the answer must follow a strict JSON schema.
Answers are checked (every field but the caveat set, none too long) and asked for once more if
they fail, because a valid schema doesn't mean sensible content.
"""

import json
import os
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

MODEL = os.environ.get('SUMMARY_MODEL') or 'xiaomi/mimo-v2.6-flash'
URL = 'https://openrouter.ai/api/v1/chat/completions'
MAX_TEXT = 120_000  # characters of the paper sent (~30k tokens): a long paper's main text fits
MAX_WORDS = 40  # a field longer than this is a broken answer (the target is 15)
KEYS = ('problem', 'approach', 'result', 'caveat')

FIELDS = {
    'problem': "What problem the paper addresses and why it matters.",
    'approach': "What the authors do about it.",
    'result': (
        "The main takeaway: what the paper shows and what the field should learn from it for future work. "
        "A number only if it is the headline."
    ),
    'caveat': (
        "The most important weakness a careful reviewer would flag, specific to this paper and shown by its own "
        "evidence: unfair or weak baselines, a tiny or synthetic evaluation, no variance reported, claims "
        "beyond the evidence, a mistake. Never speculation or a generic limitation. Empty string if nothing stands out."
    ),
}
PROMPT = (
    "You write a very quick overview of a machine learning paper. It is shown on the paper's card next to "
    "its abstract, so a researcher can tell at a glance what the paper is about; it must be much shorter than "
    "the abstract.\n\n"
    "Reply with only a JSON object with exactly these four string fields:\n"
    + "\n".join(f'- "{k}": {v}' for k, v in FIELDS.items())
    + "\n\nRules for every field: one plain sentence, target at most 15 words; don't get too technical; at most "
    "one number; faithful to the paper, only claims it makes; no hype. Someone who has never read the paper "
    "must be able to understand every field."
)
SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': list(FIELDS),
    'properties': {k: {'type': 'string', 'description': v} for k, v in FIELDS.items()},
}


class SummaryError(Exception):
    pass


def enabled():
    return os.environ.get('SUMMARIES_ENABLED', '0') == '1' and bool(os.environ.get('OPENROUTER_API_KEY'))


# -----------------------------------------------------------------------------
# the paper's text


SKIP_CLASSES = ('ltx_bibliography', 'ltx_appendix', 'ltx_page_footer', 'ltx_authors', 'ltx_role_affiliation')
SKIP_TAGS = {'script', 'style', 'nav', 'head', 'footer'}
VOID = {'br', 'img', 'hr', 'input', 'meta', 'wbr', 'col', 'source', 'link'}
BLOCK = {'p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr', 'figcaption', 'table'}


class _Text(HTMLParser):
    """the main text of a LaTeXML page: without references, appendix and authors; math as its LaTeX"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.math = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        a = dict(attrs)
        if self.skip:
            self.skip += 1
            return
        if tag in SKIP_TAGS or any(c in (a.get('class') or '').split() for c in SKIP_CLASSES):
            self.skip = 1
            return
        if self.math:
            self.math += 1
            return
        if tag == 'math':
            self.math = 1
            if a.get('alttext'):
                self.out.append(f" ${a['alttext']}$ ")
        elif tag in BLOCK:
            self.out.append('\n')

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.skip:
            self.skip -= 1
        elif self.math:
            self.math -= 1
        elif tag in BLOCK:
            self.out.append('\n')

    def handle_data(self, data):
        if not self.skip and not self.math:
            self.out.append(data)


def html_text(html):
    """the main text of arXiv's HTML version of a paper, '' if there is none"""
    start = html.find('ltx_document')
    p = _Text()
    try:
        p.feed(html[html.rfind('<', 0, start) :] if start > 0 else html)
        p.close()
    except Exception:  # untrusted input: a broken page means no text
        return ''
    text = re.sub(r'[ \t\r\f\v]+', ' ', ''.join(p.out))
    return re.sub(r'\s*\n\s*', '\n', text).strip()[:MAX_TEXT]


def pdf_text(text):
    """the text of a PDF without its references, for papers without an HTML version"""
    cut = [m.start() for m in re.finditer(r'\n\s*(References|Bibliography|REFERENCES)\s*\n', text)]
    if cut and cut[-1] > len(text) // 3:
        text = text[: cut[-1]]
    return re.sub(r'[ \t]+', ' ', text).strip()[:MAX_TEXT]


# -----------------------------------------------------------------------------
# the model


def check(answer):
    """the summary as a dict of the four fields, or SummaryError if the answer is no good"""
    try:
        j = json.loads(answer)
    except (TypeError, ValueError) as e:
        raise SummaryError(f'not JSON: {e}') from e
    if not isinstance(j, dict) or set(j) != set(KEYS) or not all(isinstance(j[k], str) for k in KEYS):
        raise SummaryError(f'wrong fields: {sorted(j) if isinstance(j, dict) else type(j).__name__}')
    j = {k: ' '.join(j[k].split()) for k in KEYS}
    for k in KEYS[:3]:
        if not j[k]:
            raise SummaryError(f'empty {k}')
    for k in KEYS:
        if len(j[k].split()) > MAX_WORDS:
            raise SummaryError(f'{k} has {len(j[k].split())} words')
    if j['caveat'].lower().strip('.') in ('empty string', 'none', 'n/a'):
        j['caveat'] = ''
    return j


def ask(title, abstract, text, model=MODEL, key=None, timeout=180):
    """the model's raw answer for one paper"""
    if text:
        user = f"Title: {title}\n\nAbstract: {abstract}\n\nFull text:\n{text}\n\n---\n\n{PROMPT}"
    else:  # made on demand for a page, or the paper's text couldn't be read
        user = (
            f"Title: {title}\n\nAbstract: {abstract}\n\n(Only the abstract is available. Work from it alone, "
            f"and leave the caveat empty unless the abstract itself shows one.)\n\n---\n\n{PROMPT}"
        )
    body = {
        'model': model,
        'messages': [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': user}],
        'max_tokens': 4000,  # room for the hidden reasoning some models do
        'response_format': {
            'type': 'json_schema',
            'json_schema': {'name': 'summary', 'strict': True, 'schema': SCHEMA},
        },
        'provider': {'require_parameters': True},  # only providers that keep to the schema
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        method='POST',
        headers={
            'Authorization': 'Bearer ' + (key or os.environ.get('OPENROUTER_API_KEY', '')),
            'Content-Type': 'application/json',
            'X-Title': 'Arxiv Resanity',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise SummaryError(f'HTTP {e.code}: {e.read()[:300].decode("utf-8", "replace")}') from e
    try:
        return d['choices'][0]['message'].get('content') or ''
    except (KeyError, IndexError, TypeError) as e:
        raise SummaryError(f'unexpected answer: {str(d)[:300]}') from e


def summarize(title, abstract, text, model=MODEL, key=None, tries=2):
    """{problem, approach, result, caveat} of a paper; asks again once if the answer is no good"""
    for i in range(tries):
        try:
            return check(ask(title, abstract, text, model, key))
        except SummaryError:
            if i == tries - 1:
                raise
