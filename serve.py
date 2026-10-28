"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

import os
import re
import json
import hashlib
from urllib.parse import urlparse
import time
import threading
from random import shuffle

import numpy as np

import click
from flask import Flask, request, redirect, url_for, abort
from flask import render_template
from flask import g  # global session-level object
from flask import session
from itsdangerous import URLSafeSerializer, URLSafeTimedSerializer, BadSignature
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash

from aslite import db, digest, mail
from aslite.db import get_papers_db, get_secret_key, load_features, features_exist, PAPERS_DB_FILE
from aslite.arxiv import find_arxiv_id, fetch_papers_by_id, OLD_ARXIV_ID_RE
from aslite.rank import rank, holdout_ranks, resolve_feats, resolve_model, has_embeddings, FEATS, MODELS

# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = 25  # number of papers to return per page
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{1,40}$')
MIN_PASSWORD = 8

app = Flask(__name__)

# signs the session cookie; see get_secret_key for where it comes from
app.secret_key = get_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',  # browsers don't send the cookie with cross-site POSTs
    SESSION_COOKIE_SECURE=os.environ.get('SECURE_COOKIES', '0') == '1',  # set when served over https
    PERMANENT_SESSION_LIFETIME=90 * 24 * 3600,
)
# behind a reverse proxy, take the client's scheme/host/ip from its X-Forwarded-* headers.
# TRUST_PROXY is the number of proxies in front; leave it 0 when clients connect directly,
# or anyone could spoof those headers
if int(os.environ.get('TRUST_PROXY', '0')) > 0:
    n = int(os.environ['TRUST_PROXY'])
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=n, x_proto=n, x_host=n, x_prefix=n)
ALLOW_SIGNUP = os.environ.get('ALLOW_SIGNUP', '1') == '1'
# where the site is reachable, for links in mails. when set, requests for any other host are
# refused, so a forged Host / X-Forwarded-Host header can't put another domain into them
PUBLIC_URL = os.environ.get('PUBLIC_URL', '').rstrip('/')
if PUBLIC_URL:
    # localhost too, for the docker healthcheck
    app.config['TRUSTED_HOSTS'] = [urlparse(PUBLIC_URL).hostname, 'localhost', '127.0.0.1']
# papers a user (and everyone together) may fetch from arXiv per day via /import, so the
# server can't be used to hammer arXiv. papers we already have don't count
FETCH_LIMIT_USER = int(os.environ.get('FETCH_LIMIT_USER', '200'))
FETCH_LIMIT_TOTAL = int(os.environ.get('FETCH_LIMIT_TOTAL', '2000'))
IMPORT_MAX_LINES = 100

# generous per-client limits on the expensive requests: (requests, per seconds). a client
# is the logged-in user, else the ip (correct behind the proxy thanks to TRUST_PROXY).
# kept in memory per worker process, so with N workers a client gets up to N times this.
# RATE_LIMITS=0 turns them off
RATE_LIMITS = {
    'rank': (120, 60),  # ranking by tags or by a paper trains a model, ~0.5-1s of cpu
    'search': (240, 60),
    'login': (30, 300),  # password hashing is deliberately slow
    'import': (60, 3600),
    'digest': (20, 3600),  # recomputing a digest trains one model per tag
    'check': (30, 3600),  # a tag check trains 5 models
    'confirm': (5, 86400),  # confirmation mails, so the email form can't be used to mail people
}
RATE_LIMITS_ON = os.environ.get('RATE_LIMITS', '1') != '0'
_buckets = {}
_buckets_lock = threading.Lock()
_last_prune = 0.0


def rate_limited(kind, client=None):
    """token bucket per (kind, client): True if this request is over the limit"""
    if not RATE_LIMITS_ON:
        return False
    n, per = RATE_LIMITS[kind]
    client = client or (('user:' + g.user) if g.user else ('ip:' + str(request.remote_addr)))
    now = time.monotonic()
    with _buckets_lock:
        global _last_prune
        if now - _last_prune > 60:
            # forget clients whose window has passed (their bucket would be full again anyway),
            # so ips and addresses are only held as long as a limit needs them
            for k in [k for k, (_, last) in _buckets.items() if now - last > RATE_LIMITS[k[0]][1]]:
                del _buckets[k]
            _last_prune = now
        tokens, last = _buckets.get((kind, client), (n, now))
        tokens = min(n, tokens + (now - last) * n / per)
        if tokens < 1:
            _buckets[(kind, client)] = (tokens, now)
            return True
        _buckets[(kind, client)] = (tokens - 1, now)
        return False


def too_many(kind):
    n, per = RATE_LIMITS[kind]
    window = f"{per // 60} minutes" if per >= 120 else f"{per} seconds"
    msg = f"Too many requests ({n} per {window} for this kind of request), please wait a bit."
    return msg, 429, {'Retry-After': str(max(1, per // n))}


# -----------------------------------------------------------------------------
# globals that manage the (lazy) loading of various state for a request


def get_tags():
    if g.user is None:
        return {}
    if not hasattr(g, '_tags'):
        g._tags = db.get_tags(g.user)
    return g._tags


def get_seen_tags():
    if g.user is None:
        return {}
    if not hasattr(g, '_seen_tags'):
        g._seen_tags = db.get_tags(g.user, neg=db.SEEN)
    return g._seen_tags


def get_neg_tags():
    if g.user is None:
        return {}
    if not hasattr(g, '_neg_tags'):
        g._neg_tags = db.get_tags(g.user, neg=True)
    return g._neg_tags


def get_papers():
    if not hasattr(g, '_pdb'):
        g._pdb = get_papers_db()
    return g._pdb


_times_cache = {}
_times_lock = threading.Lock()


def get_times():
    """
    (pid -> time, pids newest first) for all papers. reading every meta takes seconds on a
    big corpus, so each worker keeps them in memory until the papers database changes
    """
    mtime = db.db_mtime(PAPERS_DB_FILE)
    with _times_lock:
        if _times_cache.get('mtime') != mtime:
            times = get_papers().times()
            _times_cache.update(mtime=mtime, times=times, order=sorted(times, key=times.get, reverse=True))
        return _times_cache['times'], _times_cache['order']


def _session_version(user):
    """ties a session to the account's current password hash (and so to this very account)"""
    return hashlib.sha256((user['pw_hash'] or '').encode()).hexdigest()[:16]


def _start_session(user, readonly=False):
    session.clear()
    session.permanent = True
    session.update(user=user['name'], v=_session_version(user))
    if readonly:
        session['readonly'] = True


@app.before_request
def before_request():
    # everything that changes state is a POST. SameSite=Lax already keeps the session
    # cookie off cross-site POSTs; rejecting them by Sec-Fetch-Site covers old browsers too
    if request.method == 'POST' and request.headers.get('Sec-Fetch-Site') == 'cross-site':
        abort(403)

    g.user = session.get('user', None)
    g.readonly = bool(session.get('readonly'))

    if g.user:
        # a session is only as good as the account it was made for: deleting the account,
        # changing its password or a new account under the same name ends it
        u = db.get_user(g.user)
        if not u or session.get('v') != _session_version(u) or (g.readonly and not u['public']):
            session.clear()
            g.user, g.readonly = None, False

    if g.user and g.readonly:
        # a read-only view of someone's tags (login without password, if they allow it):
        # ends as soon as they stop sharing, and can't change anything
        if request.method == 'POST' and request.endpoint not in ('login', 'logout', 'confirm_email', 'unsubscribe'):
            return "error, this is a read-only view: log in with the password to make changes", 403
    elif g.user:
        # record activity on this user so we can reserve periodic
        # recommendations heavy compute only for active users
        db.touch_user(g.user)


@app.after_request
def security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    # everything comes from this server: no third party ever sees a visitor's ip (GDPR), and
    # the browser enforces it. inline scripts/styles: the templates pass data in <script>
    # blocks and KaTeX positions math with inline styles
    resp.headers.setdefault(
        'Content-Security-Policy',
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self'; connect-src 'self'; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'self'",
    )
    return resp


@app.teardown_request
def close_connection(error=None):
    # close any opened database connections
    if hasattr(g, '_pdb'):
        g._pdb.close()


# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests


def render_pid(pid, d=None):
    # render a single paper with just the information we need for the UI
    tags = get_tags()
    neg_tags = get_neg_tags()
    thumb_path = 'static/thumb/' + pid + '.jpg'
    thumb_url = thumb_path if os.path.isfile(thumb_path) else ''
    d = d or get_papers()[pid]
    return dict(
        weight=0.0,
        id=d['_id'],
        title=d['title'],
        time=d['_time_str'],
        authors=', '.join(a['name'] for a in d['authors']),
        tags=', '.join(t['term'] for t in d['tags']),
        utags=[t for t, pids in tags.items() if pid in pids],
        ntags=[t for t, pids in neg_tags.items() if pid in pids],
        stags=[t for t, pids in get_seen_tags().items() if pid in pids],
        summary=d['summary'],
        thumb_url=thumb_url,
    )


def random_rank():
    pids = list(get_times()[1])
    shuffle(pids)
    scores = [0 for _ in pids]
    return pids, scores


def time_rank():
    times, pids = get_times()
    tnow = time.time()
    scores = [(tnow - times[p]) / 60 / 60 / 24 for p in pids]  # time delta in days
    return pids, scores


def svm_rank(tags: str = '', pid: str = '', C=None, feats='auto', model='auto'):

    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or pid) or not features_exist():  # no features before compute.py's first run
        return [], [], []

    features = load_features()

    # construct the positive set, and the negative set from the user's downvotes
    pos, neg = set(), set()
    if pid:
        pos.add(pid)
    elif tags:
        tags_db, neg_db = get_tags(), get_neg_tags()
        tags_filter_to = set(tags_db.keys()) | set(neg_db.keys()) if tags == 'all' else set(tags.split(','))
        for tag, pids in tags_db.items():
            if tag in tags_filter_to:
                pos |= pids
        for tag, pids in neg_db.items():
            if tag in tags_filter_to:
                neg |= pids

    return rank(features, pos, neg, feats=feats, model=model, C=C)


def search_rank(q: str = ''):
    if not q:
        return [], []  # no query? no results
    pids, scores = get_papers().search(q)  # full-text index, see aslite/db.py
    # an arXiv url or id in the query: show that paper first
    pid = find_arxiv_id(q)
    if pid and pid in get_papers():
        pids, scores = (
            [pid] + [p for p in pids if p != pid],
            [max(scores, default=0) + 1] + [sc for p, sc in zip(pids, scores, strict=True) if p != pid],
        )
    return pids, scores


# -----------------------------------------------------------------------------
# primary application endpoints


def default_context():
    # any global context across all pages, e.g. related to the current user
    context = {}
    context['user'] = g.user if g.user is not None else ''
    context['readonly'] = bool(g.user and g.readonly)
    return context


@app.route('/', methods=['GET'])
def main():

    # default settings
    default_rank = 'time'
    default_tags = ''
    default_time_filter = ''
    default_skip_have = 'no'

    # override variables with any provided options via the interface
    opt_rank = request.args.get('rank', default_rank)  # rank type. search|tags|pid|time|random
    opt_q = request.args.get('q', '')  # search request in the text box
    opt_tags = request.args.get('tags', default_tags)  # tags to rank by if opt_rank == 'tag'
    opt_pid = request.args.get('pid', '')  # pid to find nearest neighbors to
    opt_time_filter = request.args.get('time_filter', default_time_filter)  # number of days to filter by
    opt_skip_have = request.args.get('skip_have', default_skip_have)  # hide papers we already have?
    opt_svm_c = request.args.get('svm_c', '')  # C (inverse regularization) of the ranking model, '' = default
    opt_feats = request.args.get('feats', 'auto')  # auto|tfidf|emb|concat|fusion, see aslite/rank.py
    opt_model = request.args.get('model', 'auto')  # auto|svm|logreg, see aslite/rank.py
    opt_page_number = request.args.get('page_number', '1')  # page number for pagination

    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if opt_q:
        opt_rank = 'search'

    # C from the url, within sane bounds (0, nan or 1e12 would crash the fit or hold a worker
    # for a long time); None means the tuned per-model default
    try:
        C = float(opt_svm_c)
        if not (1e-4 <= C <= 1e4):
            C = None
    except ValueError:
        C = None
    if opt_rank not in ('search', 'tags', 'pid', 'time', 'random'):
        opt_rank = default_rank
    try:
        opt_time_filter = str(min(36500, max(1, int(opt_time_filter)))) if opt_time_filter else ''
    except ValueError:
        opt_time_filter = ''
    features_ok = features_exist()
    if opt_model not in MODELS:
        opt_model = 'auto'
    if opt_feats not in FEATS or (features_ok and opt_feats != 'tfidf' and not has_embeddings(load_features())):
        opt_feats = 'auto'  # e.g. embeddings were asked for but not computed
    eff_feats = resolve_feats(load_features(), opt_feats) if features_ok else 'tfidf'
    eff_model = resolve_model(load_features(), opt_model) if features_ok else 'svm'

    # rank papers: by tags, by time, by random
    words = []  # only populated in the case of svm rank
    if opt_rank in ('tags', 'pid', 'search') and rate_limited('search' if opt_rank == 'search' else 'rank'):
        return too_many('search' if opt_rank == 'search' else 'rank')
    if opt_rank == 'search':
        pids, scores = search_rank(q=opt_q)
    elif opt_rank == 'tags':
        pids, scores, words = svm_rank(tags=opt_tags, C=C, feats=opt_feats, model=opt_model)
    elif opt_rank == 'pid':
        pids, scores, words = svm_rank(pid=opt_pid, C=C, feats=opt_feats, model=opt_model)
    elif opt_rank == 'time':
        pids, scores = time_rank()
    elif opt_rank == 'random':
        pids, scores = random_rank()
    else:
        raise ValueError("opt_rank %s is not a thing" % (opt_rank,))

    # where each paper landed among all papers, before filters: shown as "top 0.3%"
    n_ranked = len(pids)
    position = {p: i for i, p in enumerate(pids)} if opt_rank in ('tags', 'pid') else {}

    # filter by time
    if opt_time_filter:
        times = get_times()[0]
        tnow = time.time()
        deltat = int(opt_time_filter) * 60 * 60 * 24  # allowed time delta in seconds
        keep = [i for i, pid in enumerate(pids) if (tnow - times.get(pid, 0)) < deltat]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # optionally hide papers we already have (tagged or downvoted)
    if opt_skip_have == 'yes':
        have = set().union(*get_tags().values(), *get_neg_tags().values(), *get_seen_tags().values())
        keep = [i for i, pid in enumerate(pids) if pid not in have]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # crop the number of results to RET_NUM, and paginate
    try:
        page_number = max(1, int(opt_page_number))
    except ValueError:
        page_number = 1
    start_index = (page_number - 1) * RET_NUM  # desired starting index
    end_index = min(start_index + RET_NUM, len(pids))  # desired ending index
    pids = pids[start_index:end_index]
    scores = scores[start_index:end_index]

    # render all papers to just the information we need for the UI
    found = get_papers().get_many(pids)
    papers = [render_pid(pid, found[pid]) for pid in pids]
    for i, p in enumerate(papers):
        p['weight'] = float(scores[i])
        if p['id'] in position:
            p['pct'] = (position[p['id']] + 1) / n_ranked

    # build the current tags for the user, and append the special 'all' tag
    tags = get_tags()
    rtags = [{'name': t, 'n': len(pids)} for t, pids in tags.items()]
    if rtags:
        rtags.append({'name': 'all'})

    # build the page context information and render
    context = default_context()
    context['papers'] = papers
    context['tags'] = rtags
    context['words'] = words
    context['words_desc'] = (
        "Here are the top 40 most positive and bottom 20 most negative tfidf word weights of the model. "
        "If they don't look great then try tuning the regularization strength hyperparameter, svm_c, above. "
        "Lower C is higher regularization."
        + (
            " (With feats=fusion these are from the tfidf half; the embedding half has no words to show.)"
            if eff_feats == 'fusion'
            else ""
        )
    )
    context['gvars'] = {}
    context['gvars']['rank'] = opt_rank
    context['gvars']['tags'] = opt_tags
    context['gvars']['pid'] = opt_pid
    context['gvars']['time_filter'] = opt_time_filter
    context['gvars']['skip_have'] = opt_skip_have
    context['gvars']['search_query'] = opt_q
    context['gvars']['svm_c'] = '' if C is None else str(C)
    context['gvars']['feats'] = opt_feats
    context['gvars']['model'] = opt_model
    context['gvars']['eff_feats'] = eff_feats
    context['gvars']['eff_model'] = eff_model
    context['gvars']['has_emb'] = features_ok and has_embeddings(load_features())
    context['gvars']['page_number'] = str(page_number)
    return render_template('index.html', **context)


@app.route('/inspect', methods=['GET'])
def inspect():

    # fetch the paper of interest based on the pid
    pid = request.args.get('pid', '')
    pdb = get_papers()
    if pid not in pdb:
        return "error, malformed pid"  # todo: better error handling

    # load the tfidf vectors, the vocab, and the idf table
    if not features_exist():
        return "no features yet, run compute.py"
    features = load_features()
    x = features['x']
    idf = features['idf']
    ivocab = {v: k for k, v in features['vocab'].items()}
    if pid not in features['ptoi']:
        return "this paper has no features yet, they are computed with the next update"
    pix = features['ptoi'][pid]
    wixs = np.flatnonzero(np.asarray(x[pix].todense()))
    words = []
    for ix in wixs:
        words.append(
            {
                'word': ivocab[ix],
                'weight': float(x[pix, ix]),
                'idf': float(idf[ix]),
            }
        )
    words.sort(key=lambda w: w['weight'], reverse=True)

    # package everything up and render
    paper = render_pid(pid)
    context = default_context()
    context['paper'] = paper
    context['words'] = words
    context['words_desc'] = (
        "The following are the tokens and their (tfidf) weight in the paper vector. This is the actual summary that feeds into the SVM to power recommendations, so hopefully it is good and representative!"
    )
    return render_template('inspect.html', **context)


@app.route('/profile')
def profile(error=''):
    context = default_context()
    context['error'] = error
    context['allow_signup'] = ALLOW_SIGNUP
    user = db.get_user(g.user) if g.user and not g.readonly else None  # the email stays private
    context['email'] = user['email'] if user else ''
    context['pending_email'] = user['pending_email'] if user else ''
    context['email_confirmed'] = (
        time.strftime('%b %d %Y', time.localtime(user['email_confirmed'])) if user and user['email_confirmed'] else ''
    )
    context['public'] = bool(user and user['public'])
    return render_template('profile.html', **context)


@app.route('/sharing', methods=['POST'])
def sharing():
    """the user's choice whether others may view their tags read-only, without the password"""
    if g.user is None:
        return redirect(url_for('profile'))
    db.set_public(g.user, request.form.get('public') == '1')
    return redirect(url_for('profile'))


@app.route('/stats')
def stats():
    context = default_context()
    times = list(get_times()[0].values())
    tstr = lambda t: time.strftime('%b %d %Y', time.localtime(t))

    context['num_papers'] = len(times)
    if len(times) > 0:
        context['earliest_paper'] = tstr(min(times))
        context['latest_paper'] = tstr(max(times))
    else:
        context['earliest_paper'] = 'N/A'
        context['latest_paper'] = 'N/A'

    # count number of papers from various time deltas to now
    tnow = time.time()
    for thr in [1, 6, 12, 24, 48, 72, 96]:
        context['thr_%d' % thr] = len([t for t in times if t > tnow - thr * 60 * 60])

    return render_template('stats.html', **context)


@app.route('/about')
def about():
    context = default_context()
    return render_template('about.html', **context)


# -----------------------------------------------------------------------------
# tag related endpoints: add, delete tags for any paper


def _check_tag(tag):
    if g.user is None:
        return "error, not logged in"
    if tag in ('all', 'null'):
        return "error, cannot use the protected tag '%s'" % (tag,)
    if not tag or len(tag) > 100:
        return "error, tags are 1-100 characters"
    if any(c in tag for c in '/,') or not tag.isprintable():
        # '/' can't go through the /add/<pid>/<tag> routes, ',' separates tags in ?tags=
        return "error, tags can't contain '/' or ','"
    return None


@app.route('/add/<pid>/<tag>', methods=['POST'])
def add(pid=None, tag=None):
    err = _check_tag(tag)
    if err:
        return err
    if pid not in get_papers():
        return "error, unknown paper %s" % (pid,)
    db.tag_paper(g.user, pid, tag)  # replaces a downvote: a paper can't be both liked and downvoted
    print("added paper %s to tag %s for user %s" % (pid, tag, g.user))
    return "ok"


@app.route('/sub/<pid>/<tag>', methods=['POST'])
def sub(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if not db.untag_paper(g.user, pid, tag):
        return "user doesn't have paper %s in tag %s" % (pid, tag)
    return "ok removed pid %s from tag %s" % (pid, tag)


@app.route('/addneg/<pid>/<tag>', methods=['POST'])
def addneg(pid=None, tag=None):
    """downvote: this paper is NOT what the tag is about. used as an explicit negative"""
    err = _check_tag(tag)
    if err:
        return err
    if pid not in get_papers():
        return "error, unknown paper %s" % (pid,)
    db.tag_paper(g.user, pid, tag, neg=True)
    print("downvoted paper %s for tag %s for user %s" % (pid, tag, g.user))
    return "ok"


@app.route('/subneg/<pid>/<tag>', methods=['POST'])
def subneg(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if not db.untag_paper(g.user, pid, tag, neg=True):
        return "user hasn't downvoted paper %s for tag %s" % (pid, tag)
    return "ok removed downvote of pid %s for tag %s" % (pid, tag)


@app.route('/seen/<pid>/<tag>', methods=['POST'])
def seen(pid=None, tag=None):
    """neutral: seen it, no opinion. hidden from later digests, not used for training"""
    err = _check_tag(tag)
    if err:
        return err
    if pid not in get_papers():
        return "error, unknown paper %s" % (pid,)
    db.tag_paper(g.user, pid, tag, neg=db.SEEN)
    return "ok"


@app.route('/unseen/<pid>/<tag>', methods=['POST'])
def unseen(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if not db.untag_paper(g.user, pid, tag, neg=db.SEEN):
        return "paper %s isn't marked neutral for tag %s" % (pid, tag)
    return "ok"


@app.route('/del/<tag>', methods=['POST'])
def delete_tag(tag=None):
    if g.user is None:
        return "error, not logged in"

    if not db.delete_tag(g.user, tag):
        return "user does not have this tag"

    print("deleted tag %s for user %s" % (tag, g.user))
    return "ok"


@app.route('/import', methods=['GET', 'POST'])
def import_papers():
    """add papers by arXiv url or id, one per line, fetching the ones we don't have"""
    context = default_context()
    context.update(results=[], text='', tag='', error='', limit=FETCH_LIMIT_USER, max_lines=IMPORT_MAX_LINES)
    # the user's tags, suggested by the tag field (a new name creates a tag)
    context['user_tags'] = sorted(set(db.get_tags(g.user)) | set(db.get_tags(g.user, neg=True))) if g.user else []
    if request.method == 'GET' or g.user is None:
        return render_template('import.html', **context)
    if rate_limited('import'):
        return too_many('import')

    text = request.form.get('text', '')
    tag = request.form.get('tag', '').strip()
    context.update(text=text, tag=tag)
    if tag:
        err = _check_tag(tag)
        if err:
            context['error'] = err
            return render_template('import.html', **context), 400
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > IMPORT_MAX_LINES:
        context['error'] = f"At most {IMPORT_MAX_LINES} lines at a time."
        return render_template('import.html', **context), 400

    # one result per line: (input, pid or None, status)
    results, wanted = [], []
    for line in lines:
        pid = find_arxiv_id(line)
        if pid is None:
            old = OLD_ARXIV_ID_RE.search(line)
            results.append([line, None, 'old-style arXiv ids are not supported' if old else 'no arXiv id found'])
        else:
            results.append([line, pid, None])
            wanted.append(pid)
    wanted = list(dict.fromkeys(wanted))

    pdb = get_papers()
    have = set(pdb.get_many(wanted))
    missing = [p for p in wanted if p not in have]
    day_ago = time.time() - 86400
    budget = min(FETCH_LIMIT_USER - db.count_fetches(day_ago, g.user), FETCH_LIMIT_TOTAL - db.count_fetches(day_ago))
    to_fetch, over = missing[: max(0, budget)], set(missing[max(0, budget) :])
    fetched, failed = set(), set()
    for i in range(0, len(to_fetch), 100):
        batch = to_fetch[i : i + 100]
        try:
            with db.arxiv_slot():
                found = fetch_papers_by_id(batch)
        except Exception as e:  # arXiv down or rate limiting us: report it, don't retry here
            print("arxiv fetch failed: %s" % (e,))
            failed |= set(batch)
            continue
        pdb.upsert(found.values())
        for pid in batch:
            db.log_fetch(g.user, pid)  # misses count too: each one was a request
        fetched |= set(found)
    if fetched:
        db.request_compute()

    for r in results:
        pid = r[1]
        if pid is None:
            continue
        if pid in have:
            r[2] = 'already here'
        elif pid in fetched:
            r[2] = 'added from arXiv'
        elif pid in over:
            r[2] = 'over the daily fetch limit, try again tomorrow'
        elif pid in failed:
            r[2] = 'arXiv could not be reached, try again later'
        else:
            r[2] = 'not found on arXiv'
            r[1] = None
        if tag and r[1]:
            db.tag_paper(g.user, pid, tag)
    found = pdb.get_many([r[1] for r in results if r[1]])
    context['results'] = [
        {'line': line, 'pid': pid, 'status': status, 'title': found[pid]['title'] if pid in found else ''}
        for line, pid, status in results
    ]
    context['n_new'] = len(fetched)
    context['features'] = load_features()['ptoi'] if features_exist() else {}
    return render_template('import.html', **context)


# results of /tag_check per (user, tag, its papers, features version); a tag's check only
# changes when the tag or the features do
_check_cache = {}
CHECK_FOLDS = 5
MISFIT = 0.10  # a held-out like outside the top 10% doesn't fit its tag (see README for the numbers)


@app.route('/tag_check')
def tag_check():
    """how well a tag hangs together: each liked paper's rank when held out (see holdout_ranks)"""
    tag = request.args.get('tag', '')
    tags = get_tags()
    if g.user is None or tag not in tags:
        return {'error': 'no such tag'}, 404
    pos, neg = tags[tag], get_neg_tags().get(tag, set())
    if not features_exist():
        return {'error': 'no features yet'}, 400
    features = load_features()
    key = (g.user, tag, hash(frozenset(pos)), hash(frozenset(neg)), db.features_version())
    if key not in _check_cache:
        if rate_limited('check'):
            return {'error': 'too many tag checks, try again later'}, 429
        ranks = holdout_ranks(features, pos, neg, folds=CHECK_FOLDS)
        if len(_check_cache) > 1000:
            _check_cache.clear()
        _check_cache[key] = ranks
    ranks = _check_cache[key]
    if not ranks:
        return {'tag': tag, 'n': len(pos), 'min_papers': CHECK_FOLDS, 'ranks': {}}
    v = sorted(ranks.values())
    misfits = sorted((p for p in ranks if ranks[p] > MISFIT), key=ranks.get, reverse=True)
    found = get_papers().get_many(misfits)
    return {
        'tag': tag,
        'n': len(ranks),
        'median': v[len(v) // 2],
        'p90': v[min(len(v) - 1, int(0.9 * len(v)))],
        'misfit_threshold': MISFIT,
        'misfits': [{'id': p, 'title': found[p]['title'], 'rank': ranks[p]} for p in misfits if p in found],
        'ranks': ranks,
    }


DIGEST_DAYS = 7


@app.route('/digest', methods=['GET', 'POST'])
def digest_page():
    """the user's latest digest (what the email showed), with thumbs up/down per tag"""
    context = default_context()
    context.update(sections=[], digest_time='', digest_days=DIGEST_DAYS, error='')
    context['tags'] = [{'name': t} for t in get_tags()]  # suggestions in the tag editor
    if g.user is None:
        return render_template('digest.html', **context)
    if request.method == 'POST':
        # make a fresh one, e.g. before the first email or after a lot of rating
        if rate_limited('digest'):
            return too_many('digest')
        if not features_exist():
            context['error'] = "No features yet, run compute.py first."
            return render_template('digest.html', **context)
        sections = digest.compute_for(g.user, load_features(), get_times()[0], DIGEST_DAYS)
        db.save_digest(g.user, sections, DIGEST_DAYS)
        return redirect(url_for('digest_page'))

    d = db.get_digest(g.user)
    if d:
        found = get_papers().get_many(p for s in d['sections'] for p, _ in s['papers'])
        sections = []
        for s in d['sections']:
            papers = []
            for pid, score in s['papers']:
                if pid in found:
                    papers.append(dict(render_pid(pid, found[pid]), weight=score))
            sections.append({'tag': s['tag'], 'papers': papers})
        context.update(
            sections=sections,
            digest_time=time.strftime('%b %d', time.localtime(d['time'])),
            digest_days=d['days'],
        )
    return render_template('digest.html', **context)


def _unsubscribe_user(token):
    """
    the user an unsubscribe link is for, while it is still current: the token names the
    user and when their address was confirmed (see send_emails.unsubscribe_url), so it stops
    working once they unsubscribe, change the address or the account is deleted
    """
    try:
        user, confirmed = URLSafeSerializer(app.secret_key, salt='unsubscribe').loads(token)
    except (BadSignature, ValueError, TypeError):
        return None, False
    u = db.get_user(user)
    return user, bool(u and u['email_confirmed'] is not None and u['email_confirmed'] == confirmed)


# tokens travel in the query string, not the path: the access log records paths only
@app.route('/unsubscribe', methods=['GET', 'POST'])
def unsubscribe():
    """the link in every digest email; POST is also the one-click unsubscribe of RFC 8058"""
    token = request.args.get('token', '')
    user, current = _unsubscribe_user(token)
    if user is None:
        return "This unsubscribe link is invalid.", 400
    if not current:  # already unsubscribed, or the address changed since that mail
        return render_template('unsubscribe.html', state='stale', **default_context())
    if request.method == 'POST':
        db.set_email(user, '')
        return render_template('unsubscribe.html', state='done', name=user, **default_context())
    return render_template('unsubscribe.html', state='ask', name=user, token=token, **default_context())


# -----------------------------------------------------------------------------
# endpoints to log in and out


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    # every attempt counts per ip; failures also per account and ip (below), so nobody can
    # lock a user out by failing logins under their name from somewhere else
    if rate_limited('login', 'ip:' + str(request.remote_addr)):
        return profile(error="Too many login attempts, please wait a few minutes."), 429
    if not USERNAME_RE.match(username):
        return profile(error="Usernames are 1-40 letters, digits, '.', '_' or '-'."), 400

    user = db.get_user(username)
    if password == '':
        # no password: a read-only view, if this user shares their tags (off by default)
        if user and user['public']:
            _start_session(user, readonly=True)
            return redirect(url_for('main'))
        return profile(error="Enter the password. (Without one you can only view accounts that are shared.)"), 403
    legacy = user['pw_hash'] is None if user else bool(db.get_tags(username) or db.get_tags(username, neg=True))
    if legacy:
        # made before passwords existed (tags, but no password): claiming it by just picking a password would let
        # anyone take it over, so only the admin can give it one (`flask set-password`)
        return profile(error="This account has no password yet. Ask the admin of this site to set one."), 403
    if user is None:
        # a new account. with signups closed, only `flask set-password` can create accounts
        if not ALLOW_SIGNUP:
            return profile(error="Signups are closed on this server."), 403
        if len(password) < MIN_PASSWORD:
            return profile(error=f"Choose a password of at least {MIN_PASSWORD} characters."), 400
        db.set_password(username, generate_password_hash(password))
        print("registered user %s" % (username,))
    elif not check_password_hash(user['pw_hash'], password):
        if rate_limited('login', f'fail:{username}:{request.remote_addr}'):
            return profile(error="Too many login attempts, please wait a few minutes."), 429
        return profile(error="Wrong username or password."), 403

    _start_session(db.get_user(username))
    return redirect(url_for('main'))


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('profile'))


@app.cli.command('set-password')
@click.argument('username')
@click.password_option()
def set_password(username, password):
    """create a user or reset their password, e.g. when signups are closed"""
    if not USERNAME_RE.match(username):
        raise click.BadParameter('invalid username')
    db.set_password(username, generate_password_hash(password))
    print(f"password set for {username}")


# -----------------------------------------------------------------------------
# user settings and configurations


EMAIL_RE = re.compile(r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$', re.IGNORECASE)
CONFIRM_MAX_AGE = 3 * 86400


def _confirm_serializer():
    return URLSafeTimedSerializer(app.secret_key, salt='confirm-email')


@app.route('/register_email', methods=['POST'])
def register_email():
    """
    double opt-in: a new address only gets a confirmation link, and digests start once its
    owner clicks it. removing the address (empty field) takes effect at once
    """
    email = request.form.get('email', '').strip()
    if g.user is None:
        return redirect(url_for('profile'))
    if email == '':
        db.set_email(g.user, '')
        return redirect(url_for('profile'))
    if not EMAIL_RE.match(email) or len(email) > 254:
        return profile(error="That doesn't look like an email address."), 400
    # per user and per address, so nobody can make the server mail someone over and over
    if rate_limited('confirm') or rate_limited('confirm', 'to:' + email.lower()):
        return profile(error="Too many confirmation mails today, please try again tomorrow."), 429
    db.request_email(g.user, email)
    token = _confirm_serializer().dumps([g.user, email])
    # PUBLIC_URL, not the request's Host header, which a client could set to their own domain
    link = (PUBLIC_URL or request.host_url.rstrip('/')) + url_for('confirm_email', token=token)
    text = (
        f"Hi {g.user},\n\nplease confirm that you want the weekly Arxiv Resanity digest at this address:\n\n"
        f"{link}\n\nThe link works for 3 days. If you didn't ask for this, ignore this mail: "
        "without a click nothing is ever sent to you, and the address is not used for anything else.\n"
    )
    try:
        mail.send(mail.message(email, 'Confirm your email for Arxiv Resanity', text))
    except Exception as e:  # no mail server configured or reachable
        print("confirmation mail to %s failed: %s" % (email, e))
        return profile(error="The confirmation mail couldn't be sent. Please try again later."), 503
    return redirect(url_for('profile'))


@app.route('/confirm_email', methods=['GET', 'POST'])
def confirm_email():
    """the link in the confirmation mail. GET only shows a button, so link scanners that
    open every link in a mail can't confirm on the owner's behalf"""
    token = request.args.get('token', '')
    try:
        user, email = _confirm_serializer().loads(token, max_age=CONFIRM_MAX_AGE)
    except (BadSignature, ValueError, TypeError):  # BadSignature also covers expired links
        return render_template('confirm_email.html', state='invalid', **default_context()), 400
    if request.method == 'POST':
        ok = db.confirm_email(user, email)
        return render_template('confirm_email.html', state='done' if ok else 'stale', email=email, **default_context())
    return render_template('confirm_email.html', state='ask', email=email, token=token, **default_context())


@app.route('/export')
def export_data():
    """everything stored about the logged-in user, as a json download"""
    if g.user is None or g.readonly:
        return redirect(url_for('profile'))
    resp = app.response_class(json.dumps(db.export_user(g.user), indent=2), mimetype='application/json')
    resp.headers['Content-Disposition'] = f'attachment; filename="arxiv-resanity-{g.user}.json"'
    return resp


@app.route('/delete_account', methods=['POST'])
def delete_account():
    """erases the account and everything stored about it, after the password is confirmed"""
    if g.user is None:
        return redirect(url_for('profile'))
    user = db.get_user(g.user)
    if not user or not user['pw_hash'] or not check_password_hash(user['pw_hash'], request.form.get('password', '')):
        return profile(error="Wrong password, the account was not deleted."), 403
    db.delete_user(g.user)
    print("deleted user %s" % (g.user,))
    session.clear()
    return render_template('deleted.html', **dict(default_context(), user=''))


# operator details for the privacy policy and imprint, from the environment (see .env.example)
OPERATOR = {k: os.environ.get('OPERATOR_' + k.upper(), '') for k in ('name', 'address', 'email', 'hosting')}


@app.route('/privacy')
def privacy():
    return render_template('privacy.html', operator=OPERATOR, **default_context())


@app.route('/imprint')
def imprint():
    return render_template('imprint.html', operator=OPERATOR, **default_context())
