"""Tokenization and tiered target matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import WordList

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u2019']+")
_KEEP_RE = re.compile(r"[^a-z0-9']")
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'"})


def norm_token(word: str) -> str:
    """Lowercase a word and strip everything except letters, digits and apostrophes."""
    return _KEEP_RE.sub("", word.translate(_QUOTES).lower())


def strip_apostrophes(word: str) -> str:
    return word.replace("'", "")


@dataclass(frozen=True)
class Token:
    """A word in a piece of text, with its character span and ordinal."""

    text: str
    norm: str
    start: int
    end: int
    index: int


def tokenize(text: str) -> list[Token]:
    """Split text into word tokens, preserving character offsets."""
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        norm = norm_token(raw)
        if not norm:
            continue
        tokens.append(
            Token(
                text=raw,
                norm=norm,
                start=match.start(),
                end=match.end(),
                index=len(tokens),
            )
        )
    return tokens


@dataclass(frozen=True)
class Match:
    """A profanity hit inside a piece of text."""

    tokens: tuple[Token, ...]
    tier: str
    rule: str  # "exact" | "substring" | "phrase"
    key: str  # the word-list entry that fired

    @property
    def start_char(self) -> int:
        return self.tokens[0].start

    @property
    def end_char(self) -> int:
        return self.tokens[-1].end

    @property
    def first_index(self) -> int:
        return self.tokens[0].index

    @property
    def last_index(self) -> int:
        return self.tokens[-1].index

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def norm(self) -> str:
        return " ".join(t.norm for t in self.tokens)


class Matcher:
    """Matches word-list entries against text or a stream of ASR words."""

    def __init__(self, wordlist: WordList):
        self.wordlist = wordlist
        self._phrases: dict[int, dict[str, str]] = {}
        for phrase, tier in wordlist.phrases.items():
            parts = phrase.split()
            self._phrases.setdefault(len(parts), {})[phrase] = tier
        self._phrase_lengths = sorted(self._phrases, reverse=True)
        self._max_phrase = self._phrase_lengths[0] if self._phrase_lengths else 0

    # -- single token ----------------------------------------------------- #

    def match_token(self, norm: str) -> tuple[str, str, str] | None:
        """Return (tier, rule, key) if a single normalized word is a target."""
        if not norm:
            return None
        bare = strip_apostrophes(norm)
        if norm in self.wordlist.exclusions or bare in self.wordlist.exclusions:
            return None
        for candidate in (norm, bare):
            tier = self.wordlist.exact.get(candidate)
            if tier:
                return (tier, "exact", candidate)
        for key, tier in self.wordlist.substring.items():
            if key in bare:
                return (tier, "substring", key)
        return None

    def is_target(self, norm: str) -> bool:
        return self.match_token(norm) is not None

    # -- full text -------------------------------------------------------- #

    def find(self, text: str) -> list[Match]:
        """Find every target in `text`, phrases first (greedy, longest-first)."""
        return self.find_in_tokens(tokenize(text))

    def find_in_tokens(self, tokens: list[Token]) -> list[Match]:
        matches: list[Match] = []
        i = 0
        n = len(tokens)
        while i < n:
            hit = self._match_phrase_at(tokens, i)
            if hit is not None:
                match, consumed = hit
                matches.append(match)
                i += consumed
                continue
            single = self.match_token(tokens[i].norm)
            if single is not None:
                tier, rule, key = single
                matches.append(Match((tokens[i],), tier, rule, key))
            i += 1
        return matches

    def _match_phrase_at(
        self, tokens: list[Token], start: int
    ) -> tuple[Match, int] | None:
        remaining = len(tokens) - start
        for length in self._phrase_lengths:
            if length > remaining:
                continue
            window = tokens[start : start + length]
            # Hyphens are already gone from norm, so a plain join is enough.
            key = " ".join(strip_apostrophes(t.norm) for t in window)
            tier = self._phrases[length].get(key)
            if tier is not None:
                return Match(tuple(window), tier, "phrase", key), length
        return None
