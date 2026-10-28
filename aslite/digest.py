"""
The weekly digest: for each of a user's tags, the best papers that came out in the last
few days and that the user hasn't tagged or downvoted yet. send_emails.py computes and
stores it when it sends the email, and the /digest page shows the stored one, so the page
matches the email, loads without ranking anything, and is where the thumbs up/down go.
"""

import time

from aslite import db
from aslite.rank import rank


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
    (seen is {tag: pids} of the neutral ones), which may still fit another tag
    """
    sizes = sizes or {}
    seen = seen or {}
    now = now or time.time()
    liked = set().union(*tags.values())
    recent = {p for p, t in times.items() if now - t < days * 86400} - liked
    sections = []
    for tag in sorted(tags):
        n = sizes.get(tag, per_tag)
        if not tags[tag] or not recent or n <= 0:
            continue
        pids, scores, _ = rank(features, tags[tag], neg_tags.get(tag, set()), feats=feats, model=model)
        rated = neg_tags.get(tag, set()) | seen.get(tag, set())
        top = [[p, round(s, 4)] for p, s in zip(pids, scores, strict=True) if p in recent and p not in rated][:n]
        if top:
            sections.append({'tag': tag, 'papers': top})
    return sections


def compute_for(user, features, times, days=7, per_tag=PER_TAG, **kwargs):
    # papers marked neutral ("seen") count as rated: left out, but not used for training
    neg = db.get_tags(user, neg=True)
    seen = db.get_tags(user, neg=db.SEEN)
    sizes = db.get_digest_sizes(user)
    return compute(features, times, db.get_tags(user), neg, days, per_tag, seen=seen, sizes=sizes, **kwargs)
