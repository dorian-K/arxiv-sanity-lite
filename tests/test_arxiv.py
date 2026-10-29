import contextlib
import io
import urllib.error

import arxiv_daemon


def test_daemon_backs_off_and_gives_up(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(arxiv_daemon.time, 'sleep', sleeps.append)
    monkeypatch.setattr(arxiv_daemon, 'arxiv_slot', contextlib.nullcontext)
    monkeypatch.setattr(arxiv_daemon, 'parse_response', lambda r: r)

    def busy(search_query, start_index):
        calls.append(start_index)
        raise urllib.error.HTTPError('u', 503, 'busy', {'Retry-After': '120'}, io.BytesIO())

    monkeypatch.setattr(arxiv_daemon, 'get_response', busy)
    assert arxiv_daemon.fetch_page('q', 0) is None
    # a few tries only, each waiting at least as long as arxiv asked, and longer every time
    assert len(calls) == arxiv_daemon.MAX_TRIES and all(s >= 120 for s in sleeps) and sleeps == sorted(sleeps)

    # an empty page is a failure too (it used to retry it every 0.5 s, forever)
    calls.clear()
    monkeypatch.setattr(arxiv_daemon, 'get_response', lambda search_query, start_index: calls.append(1) or [])
    assert arxiv_daemon.fetch_page('q', 0) is None and len(calls) == arxiv_daemon.MAX_TRIES

    # a short page is retried once, then taken
    pages = iter([['a'] * 40, ['a'] * 40])
    monkeypatch.setattr(arxiv_daemon, 'get_response', lambda search_query, start_index: next(pages))
    assert len(arxiv_daemon.fetch_page('q', 0)) == 40


def test_venue_from_comment_or_journal_ref():
    from aslite.arxiv import venue

    assert venue('12 pages, 4 figures, Accepted at NeurIPS 2026') == 'Accepted at NeurIPS 2026'
    assert venue('To appear in TPAMI. Code: https://x.org') == 'To appear in TPAMI.'.rstrip('.')
    assert venue('Published as a conference paper at ICLR 2026') == 'Published as a conference paper at ICLR 2026'
    assert venue('', 'Nature 600, 1-5 (2026)') == 'Nature 600, 1-5 (2026)'  # the journal reference wins
    for no in (
        'Submitted to ICML 2026',
        'Under review',
        'v2 matches the published version',
        '10 pages',
        '',
        'Not accepted',
    ):
        assert venue(no) == '', no
