"""
Typo correction for search: 'difusion modles' -> 'diffusion models'.

The words come from the search index (compute.py writes the ones in at least MIN_DOCS papers
to data/search_terms.json, see db.save_terms). A query word that isn't one of them, nor the start
of one (search matches prefixes, so 'diffus' is fine), becomes the closest of them (fewest typos,
a swapped pair counts as one; then the most common) if that is far more common: a word in a
handful of papers can be a typo there too ('difusion' is in 3, 'diffusion' in 28,802), but a rare
name shouldn't be "corrected" into a common word. Typos common enough to be in the list
('netwrok', 7 papers) are fixed only one typo away from a word in KNOWN_MORE times more papers,
so real words stay ('casual' 401 vs 'causal', 'modals' 47 vs 'models'). ~5-30 ms per word.
"""

import re
import bisect
import difflib

MIN_DOCS = 5  # words in fewer papers aren't offered as corrections (and aren't in the file)
MORE_COMMON = 50  # a correction must be in this many times more papers than the word itself
RARE = 20  # a listed word in fewer papers may still be a typo...
KNOWN_MORE = 1000  # ...of a word one typo away in this many times more papers
CUTOFF = 0.7  # difflib similarity of the candidates, which are then ranked by typos


def typos(a, b):
    """edits from a to b: insertions, deletions, substitutions and swaps of neighbours"""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


class Speller:
    def __init__(self, terms):
        """terms: {word: number of papers}, the words a query may be corrected to"""
        self.docs = terms
        self.sorted = sorted(terms)
        self.buckets = {}
        for t in terms:
            self.buckets.setdefault((t[0], len(t)), []).append(t)

    def _known(self, w):
        if w in self.docs:
            return True
        i = bisect.bisect_left(self.sorted, w)
        return i < len(self.sorted) and self.sorted[i].startswith(w)

    def fix_word(self, w, docs_of=lambda w: 0):
        """the correction for one lowercase word, or None. docs_of(w): its papers, for rare words"""
        if len(w) < 4 or not w.isalpha():
            return None
        known = self._known(w)
        if known:
            docs = self.docs.get(w)
            if docs is None or docs >= RARE:  # the start of a word, or a common enough word
                return None
            need, max_typos = KNOWN_MORE * docs, 1
        else:
            need, max_typos = MORE_COMMON * max(1, docs_of(w)), 1 if len(w) < 6 else 2
        cands = [c for n in range(len(w) - 2, len(w) + 3) for c in self.buckets.get((w[0], n), ())]
        near = difflib.get_close_matches(w, cands, n=10, cutoff=CUTOFF)
        ok = [(typos(w, c), -self.docs[c], c) for c in near if c != w and self.docs[c] >= need]
        ok = [x for x in ok if x[0] <= max_typos]
        return min(ok)[2] if ok else None

    def correct(self, q, docs_of=lambda w: 0):
        """q with its typos fixed, or None if there are none"""
        changed = False

        def fix(m):
            nonlocal changed
            f = self.fix_word(m.group().lower(), docs_of)
            changed = changed or f is not None
            return f or m.group()

        out = re.sub(r'\w+', fix, q)
        return out if changed else None
