from censor.config import load_wordlist
from censor.matcher import Matcher, norm_token, tokenize


def matcher() -> Matcher:
    return Matcher(load_wordlist(tiers=["profanity", "religious", "sexual", "slurs"]))


def hits(text: str) -> list[str]:
    return [m.norm for m in matcher().find(text)]


def test_norm_token_strips_punctuation_and_case():
    assert norm_token("Fuck!") == "fuck"
    assert norm_token("hard-on") == "hardon"
    assert norm_token("\u2019Tis") == "'tis"


def test_tokenize_reports_character_spans():
    tokens = tokenize("oh fuck me")
    assert [t.norm for t in tokens] == ["oh", "fuck", "me"]
    assert (tokens[1].start, tokens[1].end) == (3, 7)


def test_substring_rule_catches_inflections():
    assert hits("that is unfuckingbelievable") == ["unfuckingbelievable"]
    assert hits("motherfucker") == ["motherfucker"]


def test_exclusions_protect_innocent_words():
    assert hits("a christian analyst passed the cocktail") == []
    assert hits("Scunthorpe assassin") == []
    assert hits("he'll know within the day") == []
    assert hits("what the hell happened") == ["hell"]


def test_phrases_match_greedily_and_longest_first():
    assert hits("you son of a bitch") == ["son of a bitch"]
    assert hits("jerk-off") == ["jerk off"]
    assert hits("jerk   off") == ["jerk off"]


def test_hyphen_and_space_variants_are_equivalent():
    assert hits("he had a hard-on") == ["hard on"]
    assert hits("he had a hardon") == ["hardon"]


def test_tier_selection_limits_matches():
    only_religious = Matcher(load_wordlist(tiers=["religious"]))
    assert [m.norm for m in only_religious.find("jesus christ that fucking hurt")] == [
        "jesus christ"
    ]


def test_allow_list_wins_over_substring():
    wordlist = load_wordlist(tiers=["profanity"])
    wordlist.exclusions.add("shitake")
    assert Matcher(wordlist).match_token("shitake") is None
