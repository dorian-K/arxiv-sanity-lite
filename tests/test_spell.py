from aslite.spell import Speller, typos

TERMS = {
    'diffusion': 28802,
    'diffusions': 40,
    'models': 291332,
    'modules': 13187,
    'modals': 47,
    'neural': 127698,
    'network': 122364,
    'netwrok': 7,
    'causal': 12894,
    'casual': 401,
    'attention': 30000,
    'graph': 25000,
}


def test_typos_become_the_closest_common_word():
    s = Speller(TERMS)
    assert s.correct('difusion modles') == 'diffusion models'
    assert s.correct('Atention') == 'attention'
    assert s.correct('grpah') == 'graph'
    assert s.correct('nueral netwrok') == 'neural network'  # swapped letters; a typo common enough to be listed
    assert typos('modles', 'models') == 1 and typos('modles', 'modules') == 1  # equally close: the commoner


def test_known_words_prefixes_and_rare_names_stay():
    s = Speller(TERMS)
    assert s.correct('diffusion models') is None
    assert s.correct('diffus') is None  # search matches it as a prefix
    assert s.correct('gpt 4o vit') is None  # short or not only letters
    assert s.correct('zebrafish') is None  # nothing close
    assert s.correct('casual modals') is None  # real words, if rarer ones
    # a word in a few papers is a typo only next to a far more common one
    assert s.correct('difusion', docs_of=lambda w: 3) == 'diffusion'
    assert s.correct('difusion', docs_of=lambda w: 1000) is None
