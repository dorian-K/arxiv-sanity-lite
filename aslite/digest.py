"""
The weekly digest: for each of a user's tags, the best papers that came out in the last
few days and that the user hasn't tagged or downvoted yet. send_emails.py computes and
stores it when it sends the email, and the /digest page shows the stored one, so the page
matches the email, loads without ranking anything, and is where the thumbs up/down go.
"""

import time

import numpy as np

from aslite import db
from aslite.rank import score


PER_TAG = 3  # papers per tag, unless the user set another number for the tag (0 leaves it out)
MAX_PER_TAG = 50


def compute(
    features,
    times,
    tags,
    neg_tags,
    days=7,
    per_tag=PER_TAG,
    feats='auto',
    model='auto',
    now=None,
    seen=None,
    sizes=None,
):
    """
    [{'tag', 'papers': [[pid, score], ...]}, ...] for every tag with recommendations.
    times is {pid: time} (PapersDB.times()); one model is trained per tag. sizes is
    {tag: papers} where the user picked another number than per_tag. left out of a tag's
    section: papers saved to any tag, and the ones marked not for / neutral for this tag
    (seen is {tag: pids} of the neutral ones), which may still fit another tag.
    a paper shows up once, under the tag it fits best: where it ranks highest among all
    papers (scores of different tags' models aren't comparable, their ranks are). the best
    fits are placed first, so a tag that loses a paper to another fills up with its next one
    """
    sizes = sizes or {}
    seen = seen or {}
    now = now or time.time()
    liked = set().union(*tags.values())
    recent = {p for p, t in times.items() if now - t < days * 86400} - liked
    pids = features['pids']
    ptoi = features.get('ptoi') or {p: i for i, p in enumerate(pids)}
    want, candidates = {}, []  # candidates: (rank fraction among all papers, tag, -score, row)
    for tag in sorted(tags):
        n = sizes.get(tag, per_tag)
        if not tags[tag] or not recent or n <= 0:
            continue
        s, _ = score(features, tags[tag], neg_tags.get(tag, set()), feats=feats, model=model)
        if s is None:
            continue
        want[tag] = n
        rated = neg_tags.get(tag, set()) | seen.get(tag, set())
        # only the recent papers get sorted, not all of them; deep enough to refill whatever
        # the other tags take
        rows = np.array([ptoi[p] for p in recent - rated if p in ptoi], dtype=np.int64)
        rows = rows[np.argsort(-s[rows], kind='stable')][: n + sum(sizes.get(t, per_tag) for t in tags)]
        ordered = np.sort(s)
        higher = len(s) - np.searchsorted(ordered, s[rows], side='right')
        candidates += [(h / len(s), tag, -float(s[r]), int(r)) for h, r in zip(higher, rows, strict=True)]
    picked, taken = {t: [] for t in want}, set()
    for _, tag, neg_score, r in sorted(candidates):
        if r not in taken and len(picked[tag]) < want[tag]:
            taken.add(r)
            picked[tag].append((neg_score, r))
    sections = []
    for tag in sorted(picked):
        top = [[pids[r], round(-neg, 4)] for neg, r in sorted(picked[tag])]
        if top:
            sections.append({'tag': tag, 'papers': top})
    return sections


def compute_for(user, features, times, days=7, per_tag=PER_TAG, **kwargs):
    # papers marked neutral ("seen") count as rated: left out, but not used for training
    neg = db.get_tags(user, neg=True)
    seen = db.get_tags(user, neg=db.SEEN)
    sizes = db.get_digest_sizes(user)
    return compute(features, times, db.get_tags(user), neg, days, per_tag, seen=seen, sizes=sizes, **kwargs)
