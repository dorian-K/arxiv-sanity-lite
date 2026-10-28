"""
The weekly digest: for each of a user's tags, the best papers that came out in the last
few days and that the user hasn't tagged or downvoted yet. send_emails.py computes and
stores it when it sends the email, and the /digest page shows the stored one, so the page
matches the email, loads without ranking anything, and is where the thumbs up/down go.
"""

import time

from aslite import db
from aslite.rank import rank


def compute(features, times, tags, neg_tags, days=7, per_tag=10, feats='auto', model='auto', now=None, skip=()):
    """
    [{'tag', 'papers': [[pid, score], ...]}, ...] for every tag with recommendations.
    times is {pid: time} (PapersDB.times()); one model is trained per tag
    """
    now = now or time.time()
    have = set().union(*tags.values(), *neg_tags.values(), skip)
    recent = {p for p, t in times.items() if now - t < days * 86400} - have
    sections = []
    for tag in sorted(tags):
        if not tags[tag] or not recent:
            continue
        pids, scores, _ = rank(features, tags[tag], neg_tags.get(tag, set()), feats=feats, model=model)
        top = [[p, round(s, 4)] for p, s in zip(pids, scores, strict=True) if p in recent][:per_tag]
        if top:
            sections.append({'tag': tag, 'papers': top})
    return sections


def compute_for(user, features, times, days=7, per_tag=10, **kwargs):
    # papers marked neutral ("seen") count as rated: left out, but not used for training
    neg = db.get_tags(user, neg=True)
    seen = set().union(*db.get_tags(user, neg=db.SEEN).values())
    return compute(features, times, db.get_tags(user), neg, days, per_tag, skip=seen, **kwargs)
