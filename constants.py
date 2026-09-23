# =============================================================================
# StorySpark — constants.py
# =============================================================================
#
# Static data used by the text-cleanup and kid-safety passes. Kept out
# of app.py so the pipeline logic reads top-to-bottom without a 40-line
# data literal interrupting it.
# =============================================================================

from __future__ import annotations

import re

_LEADING_ARTICLES: set[str] = {"a", "an", "the"}


# Words that make a story unsuitable for ages 3–10. TinyStories is
# trained on child-safe text and rarely trips these; distilgpt2
# occasionally does. Checked against the GENERATED story (post-cleanup),
# not the caption.
_KID_BLOCKLIST: set[str] = {
    # violence / harm
    "kill", "killed", "kills", "killing",
    "die", "died", "dies", "dead", "death", "dying",
    "hurt", "hurts", "hurting",
    "blood", "bloody", "wound", "wounded",
    "weapon", "gun", "knife", "sword", "bomb", "grenade",
    "war", "battle", "fight", "fighting", "fought",
    "attack", "attacked", "attacking",
    "stab", "stabbed", "shoot", "shot", "shooting",
    # fear / supernatural
    "scary", "scared", "frighten", "frightened", "afraid",
    "monster", "monsters", "ghost", "ghosts",
    "witch", "wizard", "demon", "demons", "devil", "devils",
    "nightmare", "nightmares", "terror", "horror",
    "creepy", "evil",
    # substances
    "alcohol", "beer", "wine", "drunk", "intoxicated",
    "drug", "drugs", "smoke", "smoking", "cigarette", "cigar",
    "vape", "vaping",
    # mild profanity
    "damn", "damned", "crap",
    # mature themes
    "naked", "nude", "sexy", "romance", "romantic",
}

# Matches most emoji codepoints: pictographs, symbols, dingbats,
# variation selectors, ZWJ sequences, and skin-tone modifiers.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U0000FE00-\U0000FE0F\U0001F3FB-\U0001F3FF\U0000200D\U00002B00-\U00002BFF]+",
    flags=re.UNICODE,
)

# Double quotes (straight + curly) and backticks. Apostrophes are NOT
# in this class so contractions like "it's" survive.
_QUOTE_RE = re.compile(r'["“”„«»`]')

# Filler words to strip when a sentence is cut mid-phrase.
_TRAILING_FILLER = re.compile(
    r"\b(a|an|the|and|but|or|so|because|with|for|to|of|in|on|at|"
    r"was|were|is|are|be|been|being)\s*$",
    re.IGNORECASE,
)
_DANGLING_LETTER = re.compile(r"\s+[A-Z]\.\s*$")
