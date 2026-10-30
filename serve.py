"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

import os
import sys
import re
import json
import hashlib
from urllib.parse import urlparse, quote
import time
import threading
from collections import OrderedDict
from itertools import zip_longest

import numpy as np

import click
from flask import Flask, request, redirect, url_for, abort
from flask import render_template, send_file
from flask import g  # global session-level object
from flask import session
from itsdangerous import URLSafeSerializer, URLSafeTimedSerializer, BadSignature
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash

from aslite import db, digest, mail, metrics, thumbs
from aslite.spell import Speller
from aslite.db import get_papers_db, get_secret_key, load_features, features_exist, PAPERS_DB_FILE
from aslite.arxiv import find_arxiv_id, fetch_papers_by_id, OLD_ARXIV_ID_RE
from aslite.rank import (
    fit,
    usable,
    apply,
    pct,
    fingerprint,
    holdout_ranks,
    resolve_feats,
    resolve_model,
    has_embeddings,
    FEATS,
    MODELS,
    DEFAULT_C,
)

# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = 25  # number of papers to return per page
FOR_YOU_DAYS = 30  # "For you", the main page of users with tags: the best papers of this many days
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
    'copy': (30, 3600),  # copying someone's shared tag into one's own
    'confirm': (5, 86400),  # confirmation mails, so the email form can't be used to mail people
    'agent': (300, 60),  # every call of an agent tool (agents.py), on top of its own kind
    'agent_rank': (30, 60),  # rank_like without a token: a model fit per call, per ip
    'agent_write': (120, 60),
    'oauth_register': (20, 3600),  # apps registering for OAuth (claude.ai does once per connector)
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


def error_page(title, message, code):
    """an error as a page of the site, for requests a person made (not the page's own fetches)"""
    return render_template('error.html', title=title, message=message, **default_context()), code


def too_many(kind):
    n, per = RATE_LIMITS[kind]
    window = f"{per // 60} minutes" if per >= 120 else f"{per} seconds"
    msg = f"Too many requests ({n} per {window} for this kind of request), please wait a bit."
    headers = {'Retry-After': str(max(1, per // n))}
    if request.args.get('format') == 'json' or request.endpoint == 'tag_check':
        return msg, 429, headers  # shown by the page that asked
    page, code = error_page('Slow down a little', msg, 429)
    return page, code, headers


# -----------------------------------------------------------------------------
# globals that manage the (lazy) loading of various state for a request


# these read the tags of the page's owner: the user, or whoever ?u= names on a shared page
# (see before_request). writes always go to g.user's own tags


def get_tags():
    if g.owner is None:
        return {}
    if not hasattr(g, '_tags'):
        g._tags = db.get_tags(g.owner)
    return g._tags


def get_seen_tags():
    if g.owner is None:
        return {}
    if not hasattr(g, '_seen_tags'):
        g._seen_tags = db.get_tags(g.owner, neg=db.SEEN)
    return g._seen_tags


def get_neg_tags():
    if g.owner is None:
        return {}
    if not hasattr(g, '_neg_tags'):
        g._neg_tags = db.get_tags(g.owner, neg=True)
    return g._neg_tags


def get_reads():
    if g.owner is None:
        return set()
    if not hasattr(g, '_reads'):
        g._reads = db.get_reads(g.owner)
    return g._reads


def own_page():
    """whether the page shows the user's own data to themselves (or their agent): notes are only shown then"""
    return bool(g.user) and g.owner == g.user and (g.token or not g.readonly)


def get_notes():
    if not own_page():
        return {}
    if not hasattr(g, '_notes'):
        g._notes = db.get_notes(g.user)
    return g._notes


def get_features():
    """the features, the same ones for the whole request even if compute.py swaps in new ones meanwhile"""
    if not hasattr(g, '_features'):
        g._features = load_features()
    return g._features


def get_papers():
    if not hasattr(g, '_pdb'):
        g._pdb = get_papers_db()
    return g._pdb


_times = None  # (mtime, *what get_times returns), swapped in one piece
_times_lock = threading.Lock()


def _load_times():
    pdb = get_papers_db()
    try:
        times = pdb.times()
        venues = pdb.venues()
    finally:
        pdb.close()
    pids = sorted(times, key=times.get, reverse=True)
    return times, pids, np.array([times[p] for p in pids], dtype=np.float64), venues


def get_times():
    """
    (pid -> time, pids newest first, their times as an array, pid -> venue for the papers that
    say where they were accepted or published, see aslite.arxiv.venue) for all papers. reading every meta takes seconds on a
    big corpus, so each worker keeps them in memory until the papers database changes.
    while one thread re-reads them, the others keep answering from the old ones
    """
    global _times
    mtime = db.db_mtime(PAPERS_DB_FILE)
    cached = _times
    if cached and cached[0] == mtime:
        return cached[1:]
    if not _times_lock.acquire(blocking=cached is None):
        return cached[1:]
    try:
        if not (_times and _times[0] == mtime):
            _times = (mtime, *_load_times())
        return _times[1:]
    finally:
        _times_lock.release()


_speller = (None, None)  # (mtime of the words file, Speller), see get_speller
_speller_lock = threading.Lock()


def get_speller():
    """typo correction for search (aslite/spell.py), None until compute.py wrote its words"""
    global _speller
    try:
        mtime = os.path.getmtime(db.SEARCH_TERMS_FILE)
    except OSError:
        return None
    if _speller[0] != mtime:
        with _speller_lock:
            if _speller[0] != mtime:
                terms, mtime = db.load_terms()
                _speller = (mtime, Speller(terms) if terms else None)
    return _speller[1]


def _warm_caches():
    """
    keeps this worker's features and paper times loaded, also right after the updater
    wrote new ones, so no request has to wait seconds for them (WARM_CACHES=1, docker)
    """
    while True:
        try:
            get_times()
            if features_exist():
                row_times(load_features())
            get_speller()
        except Exception as e:  # e.g. a database mid-replace: try again next round
            print("cache warmup failed: %s" % (e,))
        time.sleep(15)


if os.environ.get('WARM_CACHES') == '1':
    threading.Thread(target=_warm_caches, daemon=True, name='warm-caches').start()


def _session_version(user):
    """ties a session to the account's current password hash (and so to this very account)"""
    return hashlib.sha256((user['pw_hash'] or '').encode()).hexdigest()[:16]


def _start_session(user, readonly=False):
    session.clear()
    session.permanent = True
    session.update(user=user['name'], v=_session_version(user))
    if readonly:
        session['readonly'] = True


SHAREABLE = ('main', 'digest_page', 'tag_check')  # pages a ?u= link can show for another user
AGENT_ENDPOINTS = (
    'agents.mcp',
    'agents.mcp_me',
    'agents.api',
    'agents.api_index',
    'agents.oauth_token',
    'agents.oauth_register',
)  # agents.py: access tokens only, never the session cookie


@app.before_request
def before_request():
    g.t0 = time.perf_counter()  # for /stats, see record_timing
    # everything that changes state is a POST. SameSite=Lax already keeps the session
    # cookie off cross-site POSTs; rejecting them by Sec-Fetch-Site covers old browsers too
    if request.method == 'POST' and request.headers.get('Sec-Fetch-Site') == 'cross-site':
        abort(403)

    g.user = session.get('user', None)
    g.readonly = bool(session.get('readonly'))
    g.token = False

    # agents (agents.py) send an access token instead of a cookie: it acts as its user, read-only
    # unless it was made with write access. their endpoints ignore the cookie, so a browser
    # can't be made to call them as its logged-in user
    auth = request.headers.get('Authorization', '')
    if auth[:7].lower() == 'bearer ':
        found = db.check_token(auth[7:].strip())
        if found is None:
            return {'error': 'invalid or revoked access token'}, 401, {'WWW-Authenticate': agents.www_authenticate()}
        g.user, g.readonly, g.token = found[0], not found[1], True
    elif request.endpoint in AGENT_ENDPOINTS:
        g.user, g.readonly = None, False

    if g.user and not g.token:
        # a session is only as good as the account it was made for: deleting the account,
        # changing its password or a new account under the same name ends it
        u = db.get_user(g.user)
        if not u or session.get('v') != _session_version(u) or (g.readonly and not u['public']):
            session.clear()
            g.user, g.readonly = None, False

    if g.user and g.readonly:
        # a read-only view of someone's tags (login without password, if they allow it):
        # ends as soon as they stop sharing, and can't change anything
        if request.method == 'POST' and request.endpoint not in (
            'login',
            'logout',
            'confirm_email',
            'unsubscribe',
            *AGENT_ENDPOINTS,  # they check the token's scope per call
        ):
            if g.token:
                return "error, this access token is read-only", 403
            return "error, this is a read-only view: log in with the password to make changes", 403
    elif g.user:
        # record activity on this user so we can reserve periodic
        # recommendations heavy compute only for active users
        db.touch_user(g.user)

    # whose tags the page shows: one's own, or on a shared link (?u=<name>) the tags of a user
    # whose profile is public. someone else's page is read-only: votes and edits
    # still only ever go to g.user's own tags, and "copy to my tags" takes a tag over
    g.owner, g.viewing, g.not_shared = g.user, False, ''
    u = request.args.get('u', '')
    if u and u != g.user and request.method == 'GET' and request.endpoint in SHAREABLE:
        o = db.get_user(u) if USERNAME_RE.match(u) else None
        if o and o['public']:
            g.owner, g.viewing = u, True
        else:
            g.owner, g.viewing, g.not_shared = None, True, u


@app.url_defaults
def static_version(endpoint, values):
    """every url_for('static') carries the file's mtime, so browsers can keep it for good (see security_headers)"""
    if endpoint == 'static' and 'filename' in values:
        try:
            values['v'] = int(os.path.getmtime(os.path.join(app.static_folder, values['filename'])))
        except OSError:
            pass


@app.after_request
def record_timing(resp):
    """how long this kind of page took (aslite/metrics.py), shown on /stats; static files aren't counted"""
    if request.endpoint not in (None, 'static') and 't0' in g:
        metrics.record(g.get('metric_name') or request.endpoint, time.perf_counter() - g.t0)
    return resp


# everything comes from this server: no third party ever sees a visitor's ip (GDPR), and the
# browser enforces it. inline scripts/styles: the templates pass data in <script> blocks and
# KaTeX positions math with inline styles
CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; form-action 'self'; "
    "frame-ancestors 'none'; base-uri 'self'"
)


@app.after_request
def security_headers(resp):
    # a versioned static file (or one in a versioned vendor folder) never changes under its
    # url: no revalidation round trip per file on every page
    if request.endpoint == 'static' and resp.status_code == 200:
        if 'v' in request.args or (request.view_args or {}).get('filename', '').startswith('vendor/'):
            resp.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    resp.headers.setdefault('Content-Security-Policy', CSP)  # a page may widen it for itself (oauth_authorize)
    return resp


@app.teardown_request
def close_connection(error=None):
    # close any opened database connections
    if hasattr(g, '_pdb'):
        g._pdb.close()


# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests


# which thumbnail paper cards show (enrich_daemon.py makes both): the paper's first figure or
# its first page; the other one stands in when a paper has only that
THUMB_KIND = 'page' if os.environ.get('THUMB_KIND') == 'page' else 'figure'
THUMB_KINDS = (THUMB_KIND, 'page' if THUMB_KIND == 'figure' else 'figure')


def _thumb_license(pid):
    """the open license a paper's thumbnails were made under, '' if they may not be shown"""
    if 'refs_db' not in g:
        g.refs_db = db.get_refs_db()
    code = g.refs_db.thumb_license(pid)
    return code if thumbs.is_open(code) else ''


@app.teardown_appcontext
def _close_refs_db(exc):
    rdb = g.pop('refs_db', None)
    if rdb is not None:
        rdb.close()


def thumb_info(pid):
    """(url, license label, license url) of the thumbnail a card shows, or ('', '', '')"""
    _, kind = db.thumb_file(pid, THUMB_KINDS)
    try:
        code = _thumb_license(pid) if kind else ''
    except Exception as e:  # refs.sqlite trouble: no thumbnail, but never a broken page
        print("thumbnail license of %s: %s" % (pid, e))
        code = ''
    if not code:
        return '', '', ''
    return url_for('thumb', kind=kind, pid=pid), thumbs.license_label(code), thumbs.license_url(code)


@app.route('/thumb/<kind>/<path:pid>')
def thumb(kind, pid):
    if kind not in ('figure', 'page') or not re.fullmatch(r'[a-zA-Z.\-]*/?\d{4,7}(\.\d{4,5})?', pid):
        abort(404)  # only arXiv ids (2610.12345, cs/0701001) become file names
    path, _ = db.thumb_file(pid, (kind,), big=request.args.get('size') == 'big')
    if path is None or not _thumb_license(pid):
        abort(404)  # also a thumbnail made before its license was known not to allow it
    return send_file(os.path.abspath(path), mimetype='image/jpeg', max_age=86400)


def render_pid(pid, d=None):
    # render a single paper with just the information we need for the UI
    tags = get_tags()
    neg_tags = get_neg_tags()
    d = d or get_papers()[pid]
    t_url, t_label, t_license = thumb_info(pid)
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
        venue=d.get('venue', ''),
        read=pid in get_reads(),
        note=get_notes().get(pid, ''),
        thumb_url=t_url,
        thumb_license=t_label,  # the image is the authors' work under this license: credit it
        thumb_license_url=t_license,
    )


def _newer_than(opt_time_filter):
    """how many papers came out in the last opt_time_filter days (all without one); they come first in get_times()"""
    tarr = get_times()[2]
    if not opt_time_filter:
        return len(tarr)
    # the times are sorted newest first, so their negatives ascend
    return int(np.searchsorted(-tarr, -(time.time() - int(opt_time_filter) * 86400)))


def random_rank(opt_time_filter=''):
    """random papers of the time window (enough for the pages that get looked at, not a shuffle of all)"""
    pids = get_times()[1]
    rows = np.random.permutation(_newer_than(opt_time_filter))[:RANK_CACHE_RESULTS]
    return [pids[i] for i in rows], [0] * len(rows)


def time_rank(opt_time_filter=''):
    _, pids, tarr, _ = get_times()
    n = _newer_than(opt_time_filter)
    return pids[:n], ((time.time() - tarr[:n]) / 86400).tolist()  # time delta in days


def training_set(tags_db, neg_db, tags):
    """(liked, downvoted) papers of these tags: one, several (a,b) or 'all'"""
    names = set(tags_db) | set(neg_db) if tags == 'all' else set(tags.split(','))
    pos = set().union(*(pids for t, pids in tags_db.items() if t in names))
    neg = set().union(*(pids for t, pids in neg_db.items() if t in names))
    return pos, neg


# trained models, per process in front of models.sqlite (shared by all workers). the key names
# the exact training set and features version, so a model is only ever used for the state it
# was trained on: a vote, or new features, while a fit runs just make the next lookup a miss
MODEL_LRU = 64
_models = OrderedDict()
_models_inflight = {}  # key -> Event, set when the fit of that key is done (or failed)
_models_lock = threading.Lock()


def _drop_orphan_models(user, tags):
    """erases the models of a user or tags that were deleted while a model for them trained"""
    have = set(db.get_tags(user)) | set(db.get_tags(user, neg=True))
    if db.get_user(user) is None and not have:
        db.delete_models(user)
    elif tags != 'all':
        for t in set(tags.split(',')) - have:
            db.delete_models(user, t)


def _model_key(features, user, tags, pos, neg, feats, model, C):
    feats, model = resolve_feats(features, feats), resolve_model(features, model)
    stored = not tags.startswith('pid:')
    return (user if stored else '', tags, f'{feats}|{model}|{C}', fingerprint(pos, neg), features['_version'])


def models_ready(tags, mix, C, feats, model):
    """
    whether ranking by these tags of the page's owner trains nothing: every model it needs is
    in memory or models.sqlite (usually, after the refit that follows a vote). then the page
    ranks right away (~15 ms for 30 days) instead of sending a placeholder and a second request
    """
    features = get_features()
    tags_db, neg_db = get_tags(), get_neg_tags()
    names = (set(tags_db) if tags == 'all' else {t for t in tags.split(',') if t}) if mix == 'turns' else set()
    if len(names) < 2:  # as in _ranked: turns need two tags
        names = {tags}
    for t in names:
        pos, neg = training_set(tags_db, neg_db, t)
        key = _model_key(features, g.owner or '', t, pos, neg, feats, model, C)
        with _models_lock:
            if key in _models:
                continue
        if not (pos & features['ptoi'].keys()) or db.has_model(*key):  # nothing to train, or stored
            continue
        return False
    return True


def get_model(features, user, tags, pos, neg, feats='auto', model='auto', C=None):
    """
    the model for this training set: from memory, else models.sqlite, else trained now. if
    another thread of this worker is training the same one (a request, or the refit after a
    vote), waits for it instead of training it twice. tags is what it ranks by ('pid:<id>'
    for "similar to"; those stay in memory only). None if no liked paper has features yet
    """
    key = _model_key(features, user, tags, pos, neg, feats, model, C)
    stored = bool(key[0])
    while True:
        with _models_lock:
            m = _models.get(key)
            if m is not None:
                _models.move_to_end(key)
                return m
            done = _models_inflight.get(key)
            if done is None:
                done = _models_inflight[key] = threading.Event()
                break  # this thread trains it
        done.wait(120)  # then look again: if that fit failed, this thread takes over
    try:
        m = None
        if stored:
            try:
                m = db.get_model(*key)
            except Exception as e:  # only a cache: rank without it
                print("reading a cached model failed: %s" % (e,))
        if m is None:
            t0 = time.perf_counter()
            m, _ = fit(features, pos, neg, feats, model, C)
            metrics.record('model training', time.perf_counter() - t0)
            if m is not None and stored:
                try:
                    db.put_model(*key, m)
                    # deleted meanwhile (tag or account): its models must not stay behind. a
                    # deletion before put_model is seen here, one after it erases the row itself
                    _drop_orphan_models(user, tags)
                except Exception as e:
                    print("storing a model failed: %s" % (e,))
        if m is not None:
            with _models_lock:
                _models[key] = m
                while len(_models) > MODEL_LRU:
                    _models.popitem(last=False)
        return m
    finally:
        with _models_lock:
            del _models_inflight[key]
        done.set()


# after a vote, the models its pages will want (the tag's and 'all') are trained in the
# background, so the next page finds them ready. one thread per worker; the other worker
# finds the result in models.sqlite. off with PREFIT=0 (the tests drive prefit() themselves)
PREFIT = os.environ.get('PREFIT', '1') == '1'
PREFIT_DELAY = 1.0  # seconds after the latest vote, so a burst of votes ends in one fit
PREFIT_MAX = 200  # pending (user, tags); beyond that pages just train on demand
_prefit_pending = OrderedDict()  # (user, tags) -> time of its latest vote, oldest first
_prefit_cv = threading.Condition()
_prefit_thread = None


def prefit_later(user, *tag_sets):
    """queues the models of these tag sets of the user for training, after PREFIT_DELAY"""
    global _prefit_thread
    if not PREFIT:
        return
    with _prefit_cv:
        for tags in tag_sets:
            key = (user, tags)
            if key in _prefit_pending or len(_prefit_pending) < PREFIT_MAX:
                _prefit_pending[key] = time.time()
                _prefit_pending.move_to_end(key)
        if _prefit_thread is None or not _prefit_thread.is_alive():
            _prefit_thread = threading.Thread(target=_prefit_loop, daemon=True, name='prefit')
            _prefit_thread.start()
        _prefit_cv.notify()


def _prefit_loop():
    while True:
        with _prefit_cv:
            while True:
                if not _prefit_pending:
                    _prefit_cv.wait()
                    continue
                key, voted = next(iter(_prefit_pending.items()))
                wait = voted + PREFIT_DELAY - time.time()
                if wait <= 0:
                    del _prefit_pending[key]  # a vote from now on queues it again
                    break
                _prefit_cv.wait(wait)
        try:
            prefit(*key)
        except Exception as e:  # one failed job must not end the thread
            print("training a model in the background failed: %s" % (e,))


def prefit(user, tags):
    """
    trains (or finds) the model that a page of these tags will want. reads the user's tags now,
    not when the vote queued it, so it trains on the latest state; a vote during the fit queues
    it again, and the model of the older state is stored under that state's fingerprint
    """
    if not features_exist():
        return
    tags_db, neg_db = db.get_tags(user), db.get_tags(user, neg=True)
    if tags != 'all' and tags not in tags_db:
        return  # deleted meanwhile, or only downvotes left: nothing to train
    pos, neg = training_set(tags_db, neg_db, tags)
    get_model(load_features(), user, tags, pos, neg)


def svm_rank(tags: str = '', pid: str = '', C=None, feats='auto', model='auto'):
    """
    (model, unindexed, n_pos): the model that ranks by these tags of the page's owner (or by
    one paper, pid), None if there is nothing to rank by. _model_ranking ranks with it
    """
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or pid) or not features_exist():  # no features before compute.py's first run
        return None, 0, 0
    features = get_features()
    if pid:
        pos, neg = {pid}, set()
    else:
        pos, neg = training_set(get_tags(), get_neg_tags(), tags)
    # papers added since the last compute.py run have no features yet, so they can't count
    unindexed = sum(p not in features['ptoi'] for p in pos)
    m = get_model(features, g.owner or '', 'pid:' + pid if pid else tags, pos, neg, feats, model, C)
    return m, unindexed, len(pos)


_row_times = (None, None, None, None)  # (features, times, time of each row, venue mask), see row_times


def _rows_info(features):
    global _row_times
    times, venues = get_times()[0], get_times()[3]
    if _row_times[0] is not features or _row_times[1] is not times:
        pids = features['pids']
        _row_times = (
            features,
            times,
            np.array([times.get(p, 0) for p in pids], dtype=np.float64),
            np.array([p in venues for p in pids], dtype=bool),
        )
    return _row_times


def row_times(features):
    """the time of every paper in the order of the features' rows (0 if unknown), for numpy filters"""
    return _rows_info(features)[2]


def row_published(features):
    """whether each paper, in the order of the features' rows, says where it was accepted or published"""
    return _rows_info(features)[3]


def _model_ranking(m, opt_time_filter, skip_have, scope=None, limit=None, published=False):
    """
    (pids, scores, pct) of the papers that pass the filters (see _filtered), best first, by
    model m. only the papers of the time window get scored: compute.py stores the rows oldest
    first, so "the last 30 days" is a slice at the end (a view, no copy). pct is where each
    landed among all papers ("top 0.3%"), from the model's sketch. limit: only the best this many
    """
    features = get_features()
    n = len(features['pids'])
    if opt_time_filter:
        rows = np.flatnonzero(row_times(features) > time.time() - int(opt_time_filter) * 86400)
        contiguous = len(rows) and rows[-1] - rows[0] + 1 == len(rows)
        sel = slice(int(rows[0]), int(rows[-1]) + 1) if contiguous else rows  # older features: any order
    else:
        rows, sel = np.arange(n), None
    if not len(rows):
        return [], [], []
    s = apply(features, m, sel)
    ok = np.ones(len(rows), dtype=bool)
    if skip_have:
        ptoi = features['ptoi']
        rated = np.array([ptoi[p] for p in _rated(scope) if p in ptoi], dtype=np.int64)
        ok &= ~np.isin(rows, rated)
    if published:
        ok &= row_published(features)[rows]
    scorable = usable(features, m['feats'])  # papers still waiting for an embedding: not yet
    if scorable is not None:
        ok &= scorable[rows]
    keep = np.flatnonzero(ok)
    if limit and limit < len(keep):  # the best few first, then sort only those
        keep = np.sort(keep[np.argpartition(-s[keep], limit - 1)[:limit]])
    order = keep[np.argsort(-s[keep], kind='stable')]
    pids = features['pids']
    return [pids[r] for r in rows[order]], s[order].tolist(), pct(m, s[order])


def saved_rank(tags: str = '', neg=False):
    """the papers saved to these tags ('all' = every tag), newest first; neg: the ones marked not for them"""
    tags_db = get_neg_tags() if neg else get_tags()
    names = set(tags_db) if tags == 'all' else set(tags.split(','))
    pos = set().union(*(pids for t, pids in tags_db.items() if t in names))
    times = get_times()[0]
    pids = sorted(pos, key=lambda p: times.get(p, 0), reverse=True)
    tnow = time.time()
    return pids, [(tnow - times.get(p, tnow)) / 86400 for p in pids]


def unread_rank(tags: str = ''):
    """the papers saved to these tags that aren't marked as read, newest first"""
    pids, scores = saved_rank(tags)
    reads = get_reads()
    keep = [i for i, p in enumerate(pids) if p not in reads]
    return [pids[i] for i in keep], [scores[i] for i in keep]


def cited_rank(tags: str = ''):
    """
    (pids, scores, info): the papers on this site that the papers saved to these tags cite, cited
    by the most of them first (score = how many), newest first among equals; minus the papers
    already saved or marked not for these tags. info: how many saved papers have known
    references (enrich_daemon.py fetches them), and how many cited papers aren't on this site
    """
    tags_db, neg_db = get_tags(), get_neg_tags()
    names = set(tags_db) | set(neg_db) if tags == 'all' else set(tags.split(','))
    saved = set().union(*(pids for t, pids in tags_db.items() if t in names))
    down = set().union(*(pids for t, pids in neg_db.items() if t in names))
    with db.get_refs_db() as rdb:
        cited, known = rdb.cited_by(saved)
    times = get_times()[0]
    here = {c: by for c, by in cited.items() if c in times and c not in saved and c not in down}
    pids = sorted(here, key=lambda c: (-len(here[c]), -times[c]))
    info = {'saved': len(saved), 'known': len(known), 'elsewhere': sum(1 for c in cited if c not in times)}
    return pids, [len(here[c]) for c in pids], info


def _cited_notice(info):
    if not info['known']:
        return (
            "The references of the papers saved here aren't known yet. The server reads them from "
            "arXiv's HTML version of each paper in the background, a few hundred papers a day (if it "
            "has that turned on); papers without an HTML version can't be read."
        )
    more = f"; {info['elsewhere']} more cited papers aren't on this site" if info['elsewhere'] else ''
    return f"From the references of {info['known']} of the {info['saved']} papers saved here{more}."


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
    context['owner'] = g.owner or ''
    context['viewing'] = g.viewing
    context['not_shared'] = g.not_shared
    # whether the user's own pages can be shared by their address (sharing on, see /sharing)
    me = db.get_user(g.user) if g.user and not g.readonly and not g.viewing else None
    context['sharing'] = bool(me and me['public'])
    context['ask_public'] = bool(me and session.get('ask_public'))
    return context


RANKS = ('search', 'tags', 'saved', 'unread', 'down', 'cited', 'pid', 'time', 'random')
# training a model over every paper takes seconds on a big corpus: the page comes back at
# once with a placeholder and fetches these results in the background (?format=json)
SLOW_RANKS = ('tags', 'pid')
# results are cached per user and query, so paging, going back and reloading are instant.
# a vote changes the user's tags and with them the key; the TTL covers "last N days"
RANK_CACHE_SIZE = 200
RANK_CACHE_TTL = 600
RANK_CACHE_RESULTS = 40 * RET_NUM  # pages beyond the 40th are ranked again
_rank_cache = OrderedDict()
_rank_cache_lock = threading.Lock()


def _rank_cache_get(key):
    with _rank_cache_lock:
        hit = _rank_cache.get(key)
        if hit is None or time.time() - hit[0] > RANK_CACHE_TTL:
            return None
        _rank_cache.move_to_end(key)
        return hit[1]


def _rank_cache_put(key, value):
    with _rank_cache_lock:
        _rank_cache[key] = (time.time(), value)
        _rank_cache.move_to_end(key)
        while len(_rank_cache) > RANK_CACHE_SIZE:
            _rank_cache.popitem(last=False)


def _tags_fingerprint():
    """changes with any like, downvote or neutral of the user"""
    return hash(
        tuple((k, frozenset(v)) for d in (get_tags(), get_neg_tags(), get_seen_tags()) for k, v in sorted(d.items()))
    )


def _rated(scope=None):
    """
    papers to hide as already rated: saved to any tag, or marked not for / neutral for one
    of the tags in scope (None: any). a paper marked "not for" one tag can still be right
    for another, so it only disappears where it was rated
    """
    neg, seen = get_neg_tags(), get_seen_tags()
    names = set(neg) | set(seen) if scope is None else scope
    return set().union(*get_tags().values(), *(neg.get(t, ()) for t in names), *(seen.get(t, ()) for t in names))


def _filtered(pids, opt_time_filter, skip_have, scope=None, published=False):
    """
    indices of the papers that pass the time window and, with skip_have, aren't rated (see
    _rated); with published, only those that say where they were accepted or published
    """
    keep = range(len(pids))
    if opt_time_filter:
        times = get_times()[0]
        tnow = time.time()
        deltat = int(opt_time_filter) * 60 * 60 * 24  # allowed time delta in seconds
        keep = [i for i in keep if (tnow - times.get(pids[i], 0)) < deltat]
    if skip_have:
        have = _rated(scope)
        keep = [i for i in keep if pids[i] not in have]
    if published:
        venues = get_times()[3]
        keep = [i for i in keep if pids[i] in venues]
    return list(keep)


def _ranked_tags_in_turns(
    tag_names, opt_time_filter, opt_skip_have, C, opt_feats, opt_model, limit=None, published=False
):
    """
    several tags (or all) with mix=turns: one model per tag, and the tags take turns picking
    their next best paper. the default, one model trained on all the tags at once, is one
    ranking instead of one per tag, but with unrelated tags it likes papers halfway between
    topics. merging per-tag lists by score would let small tags crowd out big ones
    """
    lists, unindexed, n_pos = [], 0, 0
    for t in sorted(tag_names):
        m, u, n = svm_rank(tags=t, C=C, feats=opt_feats, model=opt_model)
        unindexed, n_pos = unindexed + u, n_pos + n
        if m is not None:
            pids, scores, pcts = _model_ranking(m, opt_time_filter, opt_skip_have == 'yes', {t}, limit, published)
            lists.append([(p, sc, pc, t) for p, sc, pc in zip(pids, scores, pcts, strict=True)])
    out, seen = [], set()
    for row in zip_longest(*lists):
        for item in row:
            if item is not None and item[0] not in seen:
                seen.add(item[0])
                out.append(item)
    return {
        'pids': [x[0] for x in out],
        'scores': [float(x[1]) for x in out],
        'pct': [x[2] for x in out],
        'for_tag': [x[3] for x in out],
        'words': [],
        'unindexed': unindexed,
        'n_pos': n_pos,
    }


def _ranked(
    opt_rank,
    opt_q,
    opt_tags,
    opt_pid,
    opt_time_filter,
    opt_skip_have,
    C,
    opt_feats,
    opt_model,
    opt_mix,
    opt_published=False,
    limit=None,
):
    """the ranked and filtered results of a query (model rankings: the best `limit`), see main()"""
    if opt_rank == 'tags' and opt_mix == 'turns':
        tag_names = set(get_tags()) if opt_tags == 'all' else {t for t in opt_tags.split(',') if t}
        if len(tag_names) > 1:
            return _ranked_tags_in_turns(
                tag_names, opt_time_filter, opt_skip_have, C, opt_feats, opt_model, limit, opt_published
            )
    if opt_rank in SLOW_RANKS:
        if opt_rank == 'tags':
            m, unindexed, n_pos = svm_rank(tags=opt_tags, C=C, feats=opt_feats, model=opt_model)
        else:
            m, unindexed, n_pos = svm_rank(pid=opt_pid, C=C, feats=opt_feats, model=opt_model)
        scope = {t for t in opt_tags.split(',') if t} if opt_rank == 'tags' and opt_tags != 'all' else None
        pids, scores, pcts = [], [], []
        if m is not None:
            pids, scores, pcts = _model_ranking(m, opt_time_filter, opt_skip_have == 'yes', scope, limit, opt_published)
        return {
            'pids': pids,
            'scores': scores,
            'pct': pcts,
            'for_tag': None,
            'words': m['words'] if m is not None else [],
            'unindexed': unindexed,
            'n_pos': n_pos,
        }
    cited = None
    if opt_rank == 'search':
        pids, scores = search_rank(q=opt_q)
    elif opt_rank == 'cited':
        pids, scores, cited = cited_rank(tags=opt_tags)
    elif opt_rank in ('saved', 'down'):
        pids, scores = saved_rank(tags=opt_tags, neg=opt_rank == 'down')
    elif opt_rank == 'unread':
        pids, scores = unread_rank(tags=opt_tags)
    elif opt_rank == 'time':
        pids, scores = time_rank(opt_time_filter)
        opt_time_filter = ''  # done
    else:
        pids, scores = random_rank(opt_time_filter)
        opt_time_filter = ''

    skip_have = opt_skip_have == 'yes' and opt_rank not in ('saved', 'unread', 'down', 'cited')
    if opt_time_filter or skip_have or opt_published:
        keep = _filtered(pids, opt_time_filter, skip_have, published=opt_published)
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]
    return {
        'pids': pids,
        'scores': scores,
        'pct': None,
        'for_tag': None,
        'words': [],
        'unindexed': 0,
        'n_pos': 0,
        'cited': cited,
    }


def _notice(opt_rank, opt_pid, res, features_ok):
    """why a ranking is empty or incomplete because papers are still being processed"""
    if opt_rank not in SLOW_RANKS:
        return ''
    if not features_ok:
        return (
            "The recommendation index is still being built (this happens once, after the first start). "
            "Search and the newest papers work meanwhile; recommendations show up when it is done."
        )
    if opt_rank == 'pid' and opt_pid not in get_papers():
        return "There's no paper with this id here. Paste its arXiv link on the add page to fetch it."
    if not res['unindexed']:
        return ''
    if opt_rank == 'pid':
        return (
            "This paper was added recently and isn't in the recommendation index yet. "
            "Similar papers show up after the next index update, usually within an hour."
        )
    if res['unindexed'] == res['n_pos']:
        return (
            "The papers in this tag were added recently and aren't in the recommendation index yet. "
            "Recommendations show up after the next index update, usually within an hour."
        )
    return (
        f"{res['unindexed']} of the {res['n_pos']} papers here were added recently and only count "
        "once the recommendation index is updated, usually within an hour."
    )


@app.route('/', methods=['GET'])
def main():

    # default settings
    default_rank = 'time'
    default_tags = ''
    default_time_filter = ''
    default_skip_have = 'no'
    # (format, page_number, u and published don't count: the page fetches its results as
    # /?format=json, page 2 of the default is /?page_number=2, ?u= is a shared link, and
    # published=yes a filter on top; all stay "For you")
    if not (set(request.args) - {'format', 'page_number', 'u', 'published'}) and get_tags():
        # the main use: a user with tags lands on the best recent papers for them (FOR_YOU_DAYS)
        default_rank, default_tags, default_time_filter, default_skip_have = 'tags', 'all', str(FOR_YOU_DAYS), 'yes'

    # override variables with any provided options via the interface
    opt_rank = request.args.get('rank', default_rank)  # rank type, one of RANKS
    opt_q = request.args.get('q', '')  # search request in the text box
    opt_tags = request.args.get('tags', default_tags)  # tags to rank by if opt_rank == 'tags' (or list, 'saved')
    opt_pid = request.args.get('pid', '')  # pid to find nearest neighbors to
    opt_time_filter = request.args.get('time_filter', default_time_filter)  # number of days to filter by
    opt_skip_have = request.args.get('skip_have', default_skip_have)  # hide papers we already have?
    opt_svm_c = request.args.get('svm_c', '')  # C (inverse regularization) of the ranking model, '' = default
    opt_feats = request.args.get('feats', 'auto')  # auto|tfidf|emb|concat|fusion, see aslite/rank.py
    opt_model = request.args.get('model', 'auto')  # auto|svm|logreg, see aslite/rank.py
    opt_mix = request.args.get('mix', 'one')  # several tags: one model on all of them, or one per tag taking turns
    opt_published = request.args.get('published') == 'yes'  # only papers accepted/published somewhere
    opt_page_number = request.args.get('page_number', '1')  # page number for pagination
    want_json = request.args.get('format') == 'json'  # the page fetching slow results, see SLOW_RANKS

    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if opt_q:
        # the search box sits in the same form as the ranking options, so it sends the current
        # page's filters along: a search from "For you" would only find unrated papers of the
        # last 30 days. they apply only when "search" is picked under ranking options
        if request.args.get('rank') != 'search':
            opt_time_filter, opt_skip_have = '', 'no'
        opt_rank = 'search'
    # typos fixed ('difusion' -> 'diffusion'), unless the user asked for exactly what they typed
    corrected = ''
    if opt_rank == 'search' and opt_q and request.args.get('exact') != 'yes':
        try:
            speller = get_speller()
            corrected = (speller and speller.correct(opt_q, get_papers().term_docs)) or ''
        except Exception as e:  # only a nicety: search what was typed
            print("typo correction failed: %s" % (e,))

    # C from the url, within sane bounds (0, nan or 1e12 would crash the fit or hold a worker
    # for a long time); None means the tuned per-model default
    try:
        C = float(opt_svm_c)
        if not (1e-4 <= C <= 1e4):
            C = None
    except ValueError:
        C = None
    if opt_rank not in RANKS:
        opt_rank = default_rank
    try:
        opt_time_filter = str(min(36500, max(1, int(opt_time_filter)))) if opt_time_filter else ''
    except ValueError:
        opt_time_filter = ''
    try:
        page_number = max(1, int(opt_page_number))
    except ValueError:
        page_number = 1
    features_ok = features_exist()
    if opt_model not in MODELS:
        opt_model = 'auto'
    if opt_mix not in ('one', 'turns'):
        opt_mix = 'one'
    if opt_feats not in FEATS or (features_ok and opt_feats != 'tfidf' and not has_embeddings(load_features())):
        opt_feats = 'auto'  # e.g. embeddings were asked for but not computed
    eff_feats = resolve_feats(load_features(), opt_feats) if features_ok else 'tfidf'
    eff_model = resolve_model(load_features(), opt_model) if features_ok else 'svm'

    # rank and filter, from the cache if this user asked the same recently
    query = (
        opt_rank,
        corrected or opt_q,
        opt_tags,
        opt_pid,
        opt_time_filter,
        opt_skip_have,
        C,
        opt_feats,
        opt_model,
        opt_mix,
        opt_published,
    )
    cacheable = opt_rank in ('search', 'tags', 'pid') and page_number * RET_NUM <= RANK_CACHE_RESULTS
    key = (g.owner, query, features_ok and db.features_version(), db.db_mtime(PAPERS_DB_FILE), _tags_fingerprint())
    res = _rank_cache_get(key) if cacheable else None
    pending = res is None and opt_rank in SLOW_RANKS and features_ok and not want_json
    if pending and opt_rank == 'tags':
        try:
            pending = not models_ready(opt_tags, opt_mix, C, opt_feats, opt_model)
        except Exception as e:  # only a shortcut: the placeholder path still works
            print("checking for cached models failed: %s" % (e,))
    if res is None and not pending:
        if opt_rank in ('tags', 'pid', 'search') and rate_limited('search' if opt_rank == 'search' else 'rank'):
            return too_many('search' if opt_rank == 'search' else 'rank')
        # enough for the cache, or for this page and whether there is a next one
        res = _ranked(*query, limit=max(RANK_CACHE_RESULTS, page_number * RET_NUM + 1))
        if cacheable:
            res['pids'], res['scores'] = res['pids'][:RANK_CACHE_RESULTS], res['scores'][:RANK_CACHE_RESULTS]
            for k in ('pct', 'for_tag'):
                res[k] = res[k] and res[k][:RANK_CACHE_RESULTS]
            _rank_cache_put(key, res)

    # the requested page, rendered to just the information the UI needs
    papers, has_next, words = [], False, []
    notice = _notice(opt_rank, opt_pid, res, features_ok) if res is not None else ''
    if res is not None and res.get('cited'):
        notice = _cited_notice(res['cited'])
    if res is not None:
        start_index = (page_number - 1) * RET_NUM
        pids = res['pids'][start_index : start_index + RET_NUM]
        has_next = len(res['pids']) > start_index + RET_NUM
        found = get_papers().get_many(pids)
        for i, pid in enumerate(pids, start=start_index):
            if pid not in found:
                continue  # e.g. a paper gone from the database: skip it, keeping scores aligned
            p = render_pid(pid, found[pid])
            p['weight'] = res['scores'][i]
            if res['pct']:
                p['pct'] = res['pct'][i]
            if res['for_tag']:
                p['for_tag'] = res['for_tag'][i]  # picked by this tag (see _ranked_tags_in_turns)
            papers.append(p)
        words = res['words']
    # the kind of list, for /stats: the first response of a slow one is just the placeholder
    g.metric_name = f"list:{opt_rank}" + (':results' if want_json else ':placeholder' if pending else '')
    if want_json:
        return {'papers': papers, 'words': words, 'has_next': has_next, 'notice': notice}

    # the user's tags, and 'all' for all of them together (the "For you" page)
    tags = get_tags()
    rtags = [{'name': t, 'n': len(pids)} for t, pids in sorted(tags.items())]
    if rtags:
        rtags.insert(0, {'name': 'all'})

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
    # the page about one tag (or 'all'): tabs for its recommendations and its saved papers
    tag_names = (
        [t for t in opt_tags.split(',') if t] if opt_rank in ('tags', 'saved', 'unread', 'down', 'cited') else []
    )
    context['tag_view'] = bool(tag_names) and all(t == 'all' or t in tags for t in tag_names)
    context['n_saved'] = len(saved_rank(opt_tags)[0]) if context['tag_view'] else 0
    context['n_down'] = len(saved_rank(opt_tags, neg=True)[0]) if context['tag_view'] else 0
    context['n_unread'] = len(unread_rank(opt_tags)[0]) if context['tag_view'] else 0
    try:
        context['n_cited'] = len(cited_rank(opt_tags)[0]) if context['tag_view'] else 0
    except Exception as e:  # refs.sqlite trouble must never break the page itself
        print("counting cited papers failed: %s" % (e,))
        context['n_cited'] = 0
    # what the other lists are, as their page's title
    days = f" of the last {opt_time_filter} days" if opt_time_filter else ''
    similar_to = get_papers().get_many([opt_pid]).get(opt_pid) if opt_rank == 'pid' else None
    context['heading'] = {
        'time': 'Newest papers' + (f" (last {opt_time_filter} days)" if opt_time_filter else ''),
        'random': 'Random papers' + days,
        'search': f'Search: {corrected or opt_q}',
        'pid': f"Similar to: {similar_to['title']}" if similar_to else 'Similar papers',
    }.get(opt_rank, '')
    context['gvars'] = {}
    context['gvars']['rank'] = opt_rank
    context['gvars']['tags'] = opt_tags
    context['gvars']['pid'] = opt_pid
    context['gvars']['time_filter'] = opt_time_filter
    context['gvars']['skip_have'] = opt_skip_have
    context['gvars']['search_query'] = opt_q
    context['gvars']['corrected'] = corrected
    context['gvars']['svm_c'] = '' if C is None else str(C)
    context['gvars']['feats'] = opt_feats
    context['gvars']['model'] = opt_model
    context['gvars']['mix'] = opt_mix
    context['gvars']['published'] = opt_published
    context['gvars']['eff_feats'] = eff_feats
    context['gvars']['eff_model'] = eff_model
    # the C the model uses when the field is empty (fusion trains two models, one C each)
    context['gvars']['default_c'] = (
        f"{DEFAULT_C[(eff_model, 'tfidf')]:g} / {DEFAULT_C[(eff_model, 'emb')]:g}"
        if eff_feats == 'fusion'
        else f"{DEFAULT_C[(eff_model, eff_feats)]:g}"
    )
    context['gvars']['has_emb'] = features_ok and has_embeddings(load_features())
    context['gvars']['page_number'] = str(page_number)
    context['gvars']['has_next'] = has_next
    context['gvars']['pending'] = pending
    context['gvars']['notice'] = notice
    context['for_you_days'] = FOR_YOU_DAYS
    return render_template('index.html', **context)


@app.route('/inspect', methods=['GET'])
def inspect():

    # fetch the paper of interest based on the pid
    pid = request.args.get('pid', '')
    pdb = get_papers()
    if pid not in pdb:
        return error_page(
            'Unknown paper',
            "There's no paper with this id here. Paste its arXiv link on the add page to fetch it.",
            404,
        )

    # load the tfidf vectors, the vocab, and the idf table
    if not features_exist():
        return error_page('Not ready yet', 'The index of paper words is still being built. Try again later.', 503)
    features = load_features()
    x = features['x']
    idf = features['idf']
    ivocab = {v: k for k, v in features['vocab'].items()}
    if pid not in features['ptoi']:
        return error_page(
            'Not ready yet',
            "This paper was added recently; its words are indexed with the next update, usually within an hour.",
            503,
        )
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
    sizes = db.get_digest_sizes(g.user) if user else {}
    context['digest_tags'] = (
        [{'name': t, 'n': sizes.get(t, digest.PER_TAG)} for t in sorted(get_tags())] if user else []
    )
    context['max_per_tag'] = digest.MAX_PER_TAG
    context['saved_sizes'] = request.args.get('saved') == '1'
    return render_template('profile.html', **context)


@app.route('/digest_sizes', methods=['POST'])
def digest_sizes():
    """papers per tag in the user's digest (0 = leave the tag out)"""
    if g.user is None:
        return redirect(url_for('profile'))
    tags = get_tags()
    for tag, n in zip(request.form.getlist('tag'), request.form.getlist('n'), strict=False):
        if tag not in tags:
            continue
        try:
            n = min(digest.MAX_PER_TAG, max(0, int(n)))
        except ValueError:
            continue
        db.set_digest_size(g.user, tag, None if n == digest.PER_TAG else n)
    return redirect(url_for('profile', saved=1, _anchor='digest-sizes'))


def _local_url(url):
    """a path on this site, so redirecting there can't send anyone elsewhere (no open redirect)"""
    return url.startswith('/') and not url.startswith('//') and '\\' not in url


@app.route('/sharing', methods=['POST'])
def sharing():
    """the user's choice whether others may view their tags read-only, without the password"""
    if g.user is None:
        return redirect(url_for('profile'))
    db.set_public(g.user, request.form.get('public') == '1')
    if session.pop('ask_public', None):  # answered the question after signing up: back where they were
        nxt = request.form.get('next', '')
        return redirect(nxt if _local_url(nxt) else url_for('main'))
    return redirect(url_for('profile'))


# what the kinds of pages on /stats are (aslite/metrics.py names them by endpoint or list)
STAT_LABELS = {
    'list:tags': 'Recommendations, with results right away',
    'list:tags:placeholder': 'Recommendations, placeholder first (a model to train)',
    'list:tags:results': 'Recommendations, the results after the placeholder',
    'list:pid:placeholder': 'Similar papers, placeholder first',
    'list:pid:results': 'Similar papers, the results',
    'list:pid': 'Similar papers',
    'list:search': 'Search',
    'list:time': 'Newest',
    'list:random': 'Random',
    'list:saved': 'Saved papers of a tag',
    'list:down': 'Not for this tag',
    'list:unread': 'To read',
    'agent:search': 'Agent: search',
    'agent:rank_like': 'Agent: rank like these papers',
    'agent:recommend': 'Agent: recommendations',
    'agents.mcp': 'Agent: MCP requests (all)',
    'model training': 'Training a model (after a vote, or while a page waits)',
    'add': 'Vote: save',
    'sub': 'Vote: unsave',
    'addneg': 'Vote: not for this tag',
    'subneg': 'Vote: undo not for this tag',
    'seen': 'Vote: neutral',
    'unseen': 'Vote: undo neutral',
    'digest': 'Digest',
    'inspect': 'Paper details',
    'tag_check': 'Tag check',
    'profile': 'Profile',
}


def fmt_seconds(s):
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def fmt_ago(t):
    return fmt_seconds(max(0.0, time.time() - t)) + ' ago'


@app.route('/stats')
def stats():
    context = default_context()
    tarr = get_times()[2]  # newest first
    tstr = lambda t: time.strftime('%b %d %Y', time.localtime(t))
    context['num_papers'] = len(tarr)
    context['earliest_paper'] = tstr(tarr.min()) if len(tarr) else 'N/A'
    context['latest_paper'] = tstr(tarr.max()) if len(tarr) else 'N/A'
    tnow = time.time()
    context['new_papers'] = [(h, int((tarr > tnow - h * 3600).sum())) for h in (1, 6, 12, 24, 48, 72, 96)]

    # how long pages took over the last day, every worker together (aslite/metrics.py)
    try:
        metrics.flush()
        pages = metrics.summary(tnow - 86400)
        jobs = db.get_jobs(tnow - db.METRICS_KEEP)
    except Exception as e:  # statistics only
        print("reading timings failed: %s" % (e,))
        pages, jobs = {}, []
    context['pages'] = [
        dict(
            label=STAT_LABELS.get(name, name),
            n=v['n'],
            p50=fmt_seconds(v['p50']),
            p95=fmt_seconds(v['p95']),
            max=fmt_seconds(v['max']),
        )
        for name, v in sorted(pages.items(), key=lambda kv: -kv[1]['n'])
    ]

    # background work
    runs = lambda name: [j for j in jobs if j[1] == name]
    context['computes'] = [
        dict(
            ago=fmt_ago(t),
            took=fmt_seconds(sec),
            papers=i.get('papers'),
            embedded=i.get('embedded'),
            peak=i.get('peak_mb'),
            phases=', '.join(f"{k} {fmt_seconds(v)}" for k, v in i.get('phases', {}).items()),
        )
        for t, _, sec, i in reversed(runs('compute')[-5:])
    ]
    context['fetches'] = [
        dict(ago=fmt_ago(t), took=fmt_seconds(sec), new=i.get('new'), ok=i.get('ok', True))
        for t, _, sec, i in reversed(runs('arxiv')[-5:])
    ]
    embeds = runs('embed')
    day = [j for j in embeds if j[0] > tnow - 86400]
    n_day = sum(j[3].get('papers', 0) for j in day)
    left = embeds[-1][3].get('left') if embeds else None
    context['embed'] = (
        dict(
            sessions=len(day),
            papers=n_day,
            busy=fmt_seconds(sum(j[2] for j in day)),
            per_paper=fmt_seconds(sum(j[2] for j in day) / n_day) if n_day else '',
            last=fmt_ago(embeds[-1][0]) if embeds else '',
            left=left,
            eta=f"{left / n_day:.1f} days" if left and n_day else '',
        )
        if embeds
        else None
    )
    try:
        with db.get_refs_db() as rdb:
            enrich = rdb.counts()
            enrich['titles'] = rdb.n_titles()
    except Exception as e:
        print("reading refs failed: %s" % (e,))
        enrich = None
    if enrich is not None:
        day = [j for j in runs('enrich') if j[0] > tnow - 86400]
        enrich['day'] = sum(j[3].get('papers', 0) for j in day)
        enrich['left'] = runs('enrich')[-1][3].get('left') if runs('enrich') else None
    context['enrich'] = enrich
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
    if tag in ('.', '..'):
        return "error, a tag can't be named '.' or '..'"  # browsers rewrite them in /add/<pid>/<tag> urls
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
    prefit_later(g.user, tag, 'all')
    print("added paper %s to tag %s for user %s" % (pid, tag, g.user))
    return "ok"


@app.route('/sub/<pid>/<tag>', methods=['POST'])
def sub(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if not db.untag_paper(g.user, pid, tag):
        return "user doesn't have paper %s in tag %s" % (pid, tag)
    prefit_later(g.user, tag, 'all')
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
    prefit_later(g.user, tag, 'all')
    print("downvoted paper %s for tag %s for user %s" % (pid, tag, g.user))
    return "ok"


@app.route('/subneg/<pid>/<tag>', methods=['POST'])
def subneg(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if not db.untag_paper(g.user, pid, tag, neg=True):
        return "user hasn't downvoted paper %s for tag %s" % (pid, tag)
    prefit_later(g.user, tag, 'all')
    return "ok removed downvote of pid %s for tag %s" % (pid, tag)


@app.route('/read/<pid>', methods=['POST'])
def read(pid=None):
    """the user has read this paper: it leaves the To read tab"""
    if g.user is None:
        return "error, not logged in"
    if pid not in get_papers():
        return "error, unknown paper %s" % (pid,)
    db.set_read(g.user, pid)
    return "ok"


@app.route('/note/<pid>', methods=['POST'])
def note(pid=None):
    """the user's note on a paper (empty deletes it)"""
    if g.user is None:
        return "error, not logged in"
    if pid not in get_papers():
        return "error, unknown paper %s" % (pid,)
    db.set_note(g.user, pid, request.form.get('text', ''))
    return "ok"


@app.route('/unread/<pid>', methods=['POST'])
def unread(pid=None):
    if g.user is None:
        return "error, not logged in"
    db.set_read(g.user, pid, False)
    return "ok"


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
    prefit_later(g.user, 'all')

    print("deleted tag %s for user %s" % (tag, g.user))
    return "ok"


@app.route('/copy_tag', methods=['POST'])
def copy_tag():
    """takes over a tag someone shares: its papers and downvotes, into one of the user's own tags"""
    if g.user is None:
        return redirect(url_for('profile'))
    owner, tag = request.form.get('owner', ''), request.form.get('tag', '')
    new = request.form.get('as', '').strip() or tag
    o = db.get_user(owner) if owner != g.user else None
    likes = db.get_tags(owner).get(tag) if o and o['public'] else None
    if not likes:
        return error_page('Not shared', "This tag isn't shared (anymore), or it doesn't exist.", 404)
    err = _check_tag(new)
    if err:
        return error_page("Can't use that name", err.removeprefix('error, ').capitalize() + '.', 400)
    if rate_limited('copy'):
        return too_many('copy')
    mine = db.get_tags(g.user).get(new, set())
    for pid in likes:
        db.tag_paper(g.user, pid, new)
    for pid in db.get_tags(owner, neg=True).get(tag, set()) - likes - mine:
        db.tag_paper(g.user, pid, new, neg=True)  # downvotes too, unless the user likes the paper there
    prefit_later(g.user, new, 'all')
    print("user %s copied tag %s of %s as %s" % (g.user, tag, owner, new))
    return redirect('/?rank=tags&tags=%s&skip_have=yes' % quote(new, safe=''))


def fetch_missing(wanted):
    """
    fetches the papers of `wanted` (arXiv ids) that aren't here from the arXiv API, within the
    user's and everyone's daily limits (one call per 100 ids, through db.arxiv_slot). returns
    sets: (were here, fetched now, over the limit, arXiv not reached); the rest weren't found
    """
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
    return have, fetched, over, failed


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
    # a tag is required: untagged papers would only sit in the database, found by search alone
    err = _check_tag(tag) if tag else "Pick a tag for these papers, or type a new one."
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
    have, fetched, over, failed = fetch_missing(wanted)

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
        if r[1] in have or r[1] in fetched:  # not the ones over the limit or not reached: they'd have no paper
            db.tag_paper(g.user, pid, tag)
    prefit_later(g.user, tag, 'all')
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
    if g.owner is None or tag not in tags:
        return {'error': 'no such tag'}, 404
    pos, neg = tags[tag], get_neg_tags().get(tag, set())
    if not features_exist():
        return {'error': 'no features yet'}, 400
    features = load_features()
    key = (g.owner, tag, hash(frozenset(pos)), hash(frozenset(neg)), db.features_version())
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
    if g.owner is None:
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

    d = db.get_digest(g.owner)
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
    nxt = request.form.get('next', '')
    nxt = nxt if _local_url(nxt) else ''  # e.g. back to an app's request for access (/oauth/authorize)
    if password == '':
        # no password: a read-only view, if this user's profile is public
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
    if user is None:
        # a new account starts private; the first page asks whether to make it public, with
        # the box ticked (the operator's choice, see CLAUDE.md about Art. 25(2) GDPR)
        session['ask_public'] = True
    return redirect(nxt or url_for('main'))


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


# the tools for AI agents (/mcp, /api, /llms.txt, /agents): registers its routes on app
import agents  # noqa: E402

agents.register(sys.modules[__name__])
