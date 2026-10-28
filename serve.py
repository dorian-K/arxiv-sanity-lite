"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

import os
import re
import time
import threading
from random import shuffle

import numpy as np

import click
from flask import Flask, request, redirect, url_for, abort
from flask import render_template
from flask import g  # global session-level object
from flask import session
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash

from aslite.db import get_papers_db, get_metas_db, get_tags_db, get_neg_tags_db, get_last_active_db, get_email_db
from aslite.db import get_users_db, get_secret_key, load_features, search_papers, FEATURES_FILE, PAPERS_DB_FILE
from aslite.rank import rank, resolve_feats, resolve_model, has_embeddings, FEATS, MODELS

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

# -----------------------------------------------------------------------------
# globals that manage the (lazy) loading of various state for a request


def get_tags():
    if g.user is None:
        return {}
    if not hasattr(g, '_tags'):
        with get_tags_db() as tags_db:
            tags_dict = tags_db[g.user] if g.user in tags_db else {}
        g._tags = tags_dict
    return g._tags


def get_neg_tags():
    if g.user is None:
        return {}
    if not hasattr(g, '_neg_tags'):
        with get_neg_tags_db() as neg_db:
            neg_dict = neg_db[g.user] if g.user in neg_db else {}
        g._neg_tags = neg_dict
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
    big corpus, so each worker keeps them in memory until papers.db changes
    """
    mtime = os.path.getmtime(PAPERS_DB_FILE) if os.path.isfile(PAPERS_DB_FILE) else 0
    with _times_lock:
        if _times_cache.get('mtime') != mtime:
            with get_metas_db() as mdb:
                times = {k: v['_time'] for k, v in mdb.items()}
            _times_cache.update(mtime=mtime, times=times, order=sorted(times, key=times.get, reverse=True))
        return _times_cache['times'], _times_cache['order']


@app.before_request
def before_request():
    # everything that changes state is a POST. SameSite=Lax already keeps the session
    # cookie off cross-site POSTs; rejecting them by Sec-Fetch-Site covers old browsers too
    if request.method == 'POST' and request.headers.get('Sec-Fetch-Site') == 'cross-site':
        abort(403)

    g.user = session.get('user', None)

    # record activity on this user so we can reserve periodic
    # recommendations heavy compute only for active users
    if g.user:
        with get_last_active_db(flag='c') as last_active_db:
            last_active_db[g.user] = int(time.time())


@app.after_request
def security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    return resp


@app.teardown_request
def close_connection(error=None):
    # close any opened database connections
    if hasattr(g, '_pdb'):
        g._pdb.close()


# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests


def render_pid(pid):
    # render a single paper with just the information we need for the UI
    pdb = get_papers()
    tags = get_tags()
    neg_tags = get_neg_tags()
    thumb_path = 'static/thumb/' + pid + '.jpg'
    thumb_url = thumb_path if os.path.isfile(thumb_path) else ''
    d = pdb[pid]
    return dict(
        weight=0.0,
        id=d['_id'],
        title=d['title'],
        time=d['_time_str'],
        authors=', '.join(a['name'] for a in d['authors']),
        tags=', '.join(t['term'] for t in d['tags']),
        utags=[t for t, pids in tags.items() if pid in pids],
        ntags=[t for t, pids in neg_tags.items() if pid in pids],
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
    if not (tags or pid):
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
    found = search_papers(q)  # the full-text index from compute.py
    if found is not None:
        return found
    # no index yet: scan every paper (slow on big corpora)
    qs = q.lower().strip().split()  # split query by spaces and lowercase

    pdb = get_papers()
    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs)
    matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs)
    pairs = []
    for pid, p in pdb.items():
        score = 0.0
        score += 10.0 * matchu(' '.join([a['name'] for a in p['authors']]))
        score += 20.0 * matchu(p['title'])
        score += 1.0 * match(p['summary'])
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores


# -----------------------------------------------------------------------------
# primary application endpoints


def default_context():
    # any global context across all pages, e.g. related to the current user
    context = {}
    context['user'] = g.user if g.user is not None else ''
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

    # try to parse opt_svm_c into something sensible (a float); None means "per-model default"
    try:
        C = float(opt_svm_c)
    except ValueError:
        C = None
    features_ok = os.path.isfile(FEATURES_FILE)
    if opt_model not in MODELS:
        opt_model = 'auto'
    if opt_feats not in FEATS or (features_ok and opt_feats != 'tfidf' and not has_embeddings(load_features())):
        opt_feats = 'auto'  # e.g. embeddings were asked for but not computed
    eff_feats = resolve_feats(load_features(), opt_feats) if features_ok else 'tfidf'
    eff_model = resolve_model(load_features(), opt_model) if features_ok else 'svm'

    # rank papers: by tags, by time, by random
    words = []  # only populated in the case of svm rank
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

    # filter by time
    if opt_time_filter:
        times = get_times()[0]
        tnow = time.time()
        deltat = int(opt_time_filter) * 60 * 60 * 24  # allowed time delta in seconds
        keep = [i for i, pid in enumerate(pids) if (tnow - times.get(pid, 0)) < deltat]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

    # optionally hide papers we already have (tagged or downvoted)
    if opt_skip_have == 'yes':
        have = set().union(*get_tags().values(), *get_neg_tags().values())
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
    papers = [render_pid(pid) for pid in pids]
    for i, p in enumerate(papers):
        p['weight'] = float(scores[i])

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
    features = load_features()
    x = features['x']
    idf = features['idf']
    ivocab = {v: k for k, v in features['vocab'].items()}
    pix = features['pids'].index(pid)
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
    with get_email_db() as edb:
        email = edb.get(g.user, '')
        context['email'] = email
    return render_template('profile.html', **context)


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


def _update_membership(get_db, pid, tag, present):
    """add (present=True) or remove pid from the user's tag in the given tags-like db"""
    with get_db(flag='c') as db:
        d = db[g.user] if g.user in db else {}
        if present:
            d.setdefault(tag, set()).add(pid)
        elif tag in d and pid in d[tag]:
            d[tag].remove(pid)
            if len(d[tag]) == 0:  # if this was the last paper in this tag, also delete the tag
                del d[tag]
        else:
            return d, False
        db[g.user] = d
    return d, True


def _check_tag(tag):
    if g.user is None:
        return "error, not logged in"
    if tag in ('all', 'null'):
        return "error, cannot use the protected tag '%s'" % (tag,)
    if not tag or len(tag) > 100:
        return "error, tags are 1-100 characters"
    return None


@app.route('/add/<pid>/<tag>', methods=['POST'])
def add(pid=None, tag=None):
    err = _check_tag(tag)
    if err:
        return err
    d, _ = _update_membership(get_tags_db, pid, tag, True)
    _update_membership(get_neg_tags_db, pid, tag, False)  # a paper can't be both liked and downvoted
    print("added paper %s to tag %s for user %s" % (pid, tag, g.user))
    return "ok: " + str(d)  # return back the user library for debugging atm


@app.route('/sub/<pid>/<tag>', methods=['POST'])
def sub(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    _, changed = _update_membership(get_tags_db, pid, tag, False)
    if not changed:
        return "user doesn't have paper %s in tag %s" % (pid, tag)
    return "ok removed pid %s from tag %s" % (pid, tag)


@app.route('/addneg/<pid>/<tag>', methods=['POST'])
def addneg(pid=None, tag=None):
    """downvote: this paper is NOT what the tag is about. used as an explicit negative"""
    err = _check_tag(tag)
    if err:
        return err
    d, _ = _update_membership(get_neg_tags_db, pid, tag, True)
    _update_membership(get_tags_db, pid, tag, False)
    print("downvoted paper %s for tag %s for user %s" % (pid, tag, g.user))
    return "ok: " + str(d)


@app.route('/subneg/<pid>/<tag>', methods=['POST'])
def subneg(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    _, changed = _update_membership(get_neg_tags_db, pid, tag, False)
    if not changed:
        return "user hasn't downvoted paper %s for tag %s" % (pid, tag)
    return "ok removed downvote of pid %s for tag %s" % (pid, tag)


@app.route('/del/<tag>', methods=['POST'])
def delete_tag(tag=None):
    if g.user is None:
        return "error, not logged in"

    with get_tags_db(flag='c') as tags_db:
        if g.user not in tags_db:
            return "user does not have a library"

        d = tags_db[g.user]

        if tag not in d:
            return "user does not have this tag"

        # delete the tag
        del d[tag]

        # write back to database
        tags_db[g.user] = d

    # and its downvotes, if any
    with get_neg_tags_db(flag='c') as neg_db:
        if g.user in neg_db and tag in neg_db[g.user]:
            nd = neg_db[g.user]
            del nd[tag]
            neg_db[g.user] = nd

    print("deleted tag %s for user %s" % (tag, g.user))
    return "ok: " + str(d)  # return back the user library for debugging atm


# -----------------------------------------------------------------------------
# endpoints to log in and out


def _known_user(username):
    """has this user ever stored anything (from before accounts had passwords)?"""
    for get_db in (get_tags_db, get_neg_tags_db, get_email_db):
        with get_db() as db:
            if username in db:
                return True
    return False


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    if not USERNAME_RE.match(username):
        return profile(error="Usernames are 1-40 letters, digits, '.', '_' or '-'."), 400

    with get_users_db(flag='c') as udb:
        pw_hash = udb.get(username)
        if pw_hash is None:
            # a new account, or one made before passwords existed: the first password set
            # claims it. with signups closed, only `flask set-password` can create accounts
            if not ALLOW_SIGNUP:
                return profile(error="Signups are closed on this server."), 403
            if len(password) < MIN_PASSWORD:
                return profile(error=f"Choose a password of at least {MIN_PASSWORD} characters."), 400
            udb[username] = generate_password_hash(password)
            print("registered user %s%s" % (username, " (existing data)" if _known_user(username) else ""))
        elif not check_password_hash(pw_hash, password):
            return profile(error="Wrong username or password."), 403

    session.clear()
    session.permanent = True
    session['user'] = username
    return redirect(url_for('profile'))


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
    with get_users_db(flag='c') as udb:
        udb[username] = generate_password_hash(password)
    print(f"password set for {username}")


# -----------------------------------------------------------------------------
# user settings and configurations


@app.route('/register_email', methods=['POST'])
def register_email():
    email = request.form['email']

    if g.user:
        # do some basic input validation
        proper_email = re.match(r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$', email, re.IGNORECASE)
        if email == '' or proper_email:  # allow empty email, meaning no email
            # everything checks out, write to the database
            with get_email_db(flag='c') as edb:
                edb[g.user] = email

    return redirect(url_for('profile'))
