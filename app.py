# =============================================================================
# 📖✨ StorySpark — A Storytelling Application for Kids (ages 3–10) ✨📖
# =============================================================================
#
#  Course : ISOM5240 — Deep Learning Business Applications with Python
#  Author : Elva
#  Stack  : Streamlit · Hugging Face Transformers · gTTS
#
#  Pipeline
#  --------
#    [Image upload]
#        │  1. img2text()  →  Hugging Face pipeline("image-to-text")
#        │                   (Salesforce/blip-image-captioning-base)
#        ▼
#    2. text2story()  →  Hugging Face pipeline("text-generation")
#                        (roneneldan/TinyStories-33M, fallback: distilgpt2)
#        │
#        ▼
#    3. text2audio()  →  gTTS (Google Text-to-Speech)
#        │
#        ▼
#    [Storybook spread with audio player]   🎧
#
#  All styling lives in ./style.css.
# =============================================================================

from __future__ import annotations

import base64
import io
import logging
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from gtts import gTTS

# `gTTSError` lives in different modules across gTTS releases; import it
# defensively so a future version bump doesn't crash the whole app.
try:
    from gtts import gTTSError                     # type: ignore[attr-defined]
except ImportError:                                # pragma: no cover
    from gtts.tts import gTTSError                 # type: ignore[attr-defined]

# Exceptions that indicate the user's network / DNS is unavailable. We
# treat them as a single "you're offline" bucket in the UI so the
# message stays kid-friendly.
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
)


# ---------------------------------------------------------------------------
# 0. Logging & page config
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="StorySpark — Make a Story!",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css(
    *file_names: str,
    base_dir: str | Path = "styles",
) -> None:
    """Inject the project's kid-friendly stylesheets in load order.

    Resolves every ``file_name`` relative to ``base_dir`` (default
    ``"styles"``) inside the directory that contains ``app.py``. Using
    :mod:`pathlib` keeps the resolution reliable on Streamlit Cloud,
    where the working directory may differ from the app's source
    directory.

    Later files override earlier ones on selector ties, so the chain
    MUST be::

        style.css → components.css → animations.css → responsive.css
    """
    styles_root = Path(__file__).parent / base_dir
    for file_name in file_names:
        css_path = styles_root / file_name
        if css_path.exists():
            st.markdown(
                f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True,
            )
        else:
            logger.warning("CSS file not found, skipping: %s", css_path)


load_css(
    "style.css",
    "components.css",
    "animations.css",
    "responsive.css",
)


def _html(content: str) -> None:
    """Render raw HTML via Streamlit (shorthand for st.markdown + unsafe)."""
    st.markdown(content, unsafe_allow_html=True)


def _page_heading(emoji: str, text: str) -> None:
    """Render a kid-friendly section heading."""
    _html(
        f'<div class="page-heading">'
        f'<span class="page-emoji">{emoji}</span>'
        f'<span>{text}</span></div>'
    )


# ---------------------------------------------------------------------------
# 1. Model configuration & constants
# ---------------------------------------------------------------------------

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
KID_STORY_MODEL = "roneneldan/TinyStories-33M"
FALLBACK_STORY_MODEL = "distilgpt2"

# Story-gen prompts per model. TinyStories only needs a seed sentence;
# distilgpt2 needs an explicit instruction to write a kid's bedtime story.
_STORY_PROMPTS: dict[str, str] = {
    KID_STORY_MODEL: "Once upon a time there was a {subject}. ",
    FALLBACK_STORY_MODEL: (
        "Tell a happy bedtime story for a 6 year old child about {subject}. "
        "Use simple words, short sentences, and end with something nice. "
        "Story:\n"
    ),
}
_FALLBACK_PROMPT: str = _STORY_PROMPTS[FALLBACK_STORY_MODEL]

# Maps the pipeline phase → which quest-path step should be highlighted.
#   idle    → step 1 (pick a picture)
#   working → step 2 (make my story)
#   done    → step 3 (listen & save)
PHASE_TO_ACTIVE_STEP: dict[str, int] = {"idle": 1, "working": 2, "done": 3}

# Default values for every key we put in st.session_state.
DEFAULT_SESSION_STATE: dict[str, Any] = {
    "story": "",
    "audio_bytes": b"",
    "caption": "",
    "celebrated": False,
    "phase": "idle",          # one of: idle | working | done
    "progress_step": 0,
}

# Assignment requirement: stories must be 50–100 words. The generator
# trims to (high − 7) so the canonical "happily ever after" closer
# still fits; _validate_word_count warns the user if anything slipped
# through the cracks.
STORY_WORD_COUNT_LOW: int = 50
STORY_WORD_COUNT_HIGH: int = 100
STORY_WORD_COUNT_TRIM_HIGH: int = STORY_WORD_COUNT_HIGH - 7

# Steps shown by the inline turning-book loader. Each tuple is
# (emoji, fractional progress, status text). Index matches
# st.session_state.progress_step (0..3).
LOADER_STEPS: list[tuple[str, float, str]] = [
    ("🔎", 0.15, "Looking at your picture"),
    ("📖", 0.45, "Writing your story"),
    ("🎤", 0.80, "Recording the voice"),
    ("✅", 1.00, "All done!"),
]


# ---------------------------------------------------------------------------
# 2. Cached Hugging Face pipelines
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_caption_pipeline() -> Any:
    """Lazy-load the BLIP image-to-text pipeline (cached across sessions).

    Tries the high-level ``pipeline()`` first. On failure (usually a
    missing torchvision / sentencepiece in the host environment),
    falls back to loading the processor + model directly and wrapping
    them in a tiny callable that mimics the pipeline output shape.

    Note: ``transformers.pipeline`` has ~30 task-specific overloads
    and the ``"image-to-text"`` task is not fully covered by the type
    stubs. The ``# type: ignore[reportArgumentType]`` comment on the
    ``task=`` line is the standard, accepted way to silence Pylance here.
    """
    try:
        return pipeline(
            task="image-to-text",  # type: ignore[reportArgumentType]
            model=CAPTION_MODEL,
        )
    except Exception as exc:                       # noqa: BLE001
        logger.warning(
            "pipeline('image-to-text') failed (%s); "
            "falling back to explicit model + processor load.",
            exc,
        )

    from transformers import BlipProcessor, BlipForConditionalGeneration

    try:
        processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
        model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
    except Exception as exc:                       # noqa: BLE001
        raise RuntimeError(
            f"Could not load BLIP captioner '{CAPTION_MODEL}'. "
            f"Ensure torchvision and sentencepiece are installed. "
            f"Original error: {exc}"
        ) from exc

    class _BlipCaptioner:
        """Minimal drop-in for ``pipeline('image-to-text')``.

        Explicit ``__init__`` binding keeps Pylance happy: passing the
        processor and model as constructor arguments means they're
        typed as ``Any`` (rather than unresolved closure variables),
        so ``self._processor(images=..., return_tensors=...)``
        type-checks cleanly.
        """

        def __init__(self, processor: Any, model: Any) -> None:
            self._processor = processor
            self._model = model

        def __call__(
            self,
            image: Any,
            max_new_tokens: int = 40,
        ) -> list[dict[str, str]]:
            inputs = self._processor(images=image, return_tensors="pt")
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
            text = self._processor.decode(
                output_ids[0], skip_special_tokens=True
            )
            return [{"generated_text": text}]

    return _BlipCaptioner(processor=processor, model=model)


@st.cache_resource(show_spinner=False)
def _get_story_pipeline() -> tuple[Any, str]:
    """Load TinyStories (preferred) with distilgpt2 as a fallback.

    Returns
    -------
    tuple[Any, str]
        ``(text_generation_pipeline, model_id)`` of whichever model
        loaded first. Raises ``RuntimeError`` if none succeed.
    """
    for model_id in (KID_STORY_MODEL, FALLBACK_STORY_MODEL):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            text_gen = pipeline(  # type: ignore[call-overload]
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            return text_gen, model_id
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not load %s: %s", model_id, exc)
            continue
    raise RuntimeError("No story-generation model could be loaded.")


# ---------------------------------------------------------------------------
# 3. Core pipeline: image → caption → story → audio
# ---------------------------------------------------------------------------

def img2text(url_or_image: str | Image.Image) -> str:
    """Image → short caption using the BLIP image-to-text pipeline.

    Parameters
    ----------
    url_or_image : str | PIL.Image.Image
        Either a PIL Image (typical case from the Streamlit uploader)
        or a URL string pointing at a remote image.

    Returns
    -------
    str
        A capitalised, stripped caption. Empty string on decoding failure.
    """
    captioner = _get_caption_pipeline()

    if isinstance(url_or_image, Image.Image):
        image = (
            url_or_image
            if url_or_image.mode == "RGB"
            else url_or_image.convert("RGB")
        )
    else:
        with urllib.request.urlopen(str(url_or_image)) as response:
            image = Image.open(BytesIO(response.read())).convert("RGB")

    results = captioner(image, max_new_tokens=40)
    text = (results[0].get("generated_text", "") if results else "").strip()
    return text[0].upper() + text[1:] if text else text


def text2story(caption: str) -> str:
    """Caption → kid-friendly bedtime story (50–100 words).

    Emojis and stray quotes are stripped, the ending is repaired so the
    story finishes on a complete sentence, and the result is trimmed to
    the assignment's 50–100 word target.
    """
    story_gen, model_id = _get_story_pipeline()
    prompt = _STORY_PROMPTS.get(model_id, _FALLBACK_PROMPT).format(
        subject=_extract_subject(caption),
    )

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": 160,
        "do_sample": True,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
    tok = getattr(story_gen, "tokenizer", None)
    if tok is not None and getattr(tok, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tok.pad_token_id

    outputs = story_gen(prompt, **gen_kwargs)
    generated = str(outputs[0].get("generated_text", "")) if outputs else ""

    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    generated = re.sub(r"\s+", " ", generated).strip()
    generated = _strip_emojis(generated)              # strip early
    generated = _repair_story_end(generated)
    # Truncate FIRST (leaves ~7 words of room) so the canonical happy
    # ending added below is never dropped by the word-count cap.
    generated = _truncate_to_word_count(
        generated,
        low=STORY_WORD_COUNT_LOW,
        high=STORY_WORD_COUNT_TRIM_HIGH,
    )
    generated = _ensure_happy_ending(generated)
    if not _is_safe_for_kids(generated):
        # The distilgpt2 fallback occasionally emits age-inappropriate
        # words; ship a vetted safe story instead of the model output.
        generated = _safe_fallback_story(caption)
    return _strip_emojis(generated)                   # strip final


def text2audio(story_text: str) -> bytes:
    """Story text → MP3 audio bytes via Google Text-to-Speech.

    Emojis and quotes are stripped first so gTTS never tries to
    pronounce them. Returns empty bytes if the input is empty.
    """
    if not story_text:
        return b""
    clean_text = _strip_emojis(story_text)
    if not clean_text:
        return b""

    tts = gTTS(text=clean_text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# 3a. Small text helpers
# ---------------------------------------------------------------------------

_LEADING_ARTICLES = {"a", "an", "the"}
_HAPPY_ENDING = " And they all lived happily ever after."

# Varied closers used only when a story lands under the 50-word minimum.
# Kept short, natural, and non-repeating so the padding never looks broken.
# The last entry is intentionally long so a very short (~30-word) story
# still reaches the assignment's 50-word floor.
_EXTRA_CLOSERS: list[str] = [
    " Everyone smiled and clapped.",
    " What a wonderful day it was!",
    " The end.",
    " And they played together every day, sharing toys, snacks, and big "
    "warm hugs. And they all lived happily ever after.",
]


# Word blocklist for the kid-safety pass. TinyStories is trained on
# child-safe text and never trips these; distilgpt2 occasionally does.
# We check the GENERATED story (after all cleanup), not the caption.
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
    # mature themes (avoid for ages 3-10)
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

# Double quotes (straight + curly) and backticks. Apostrophes (') are
# NOT in this class — we want to preserve contractions like "it's".
_QUOTE_RE = re.compile(r'["“”„«»`]')


def _strip_emojis(text: str) -> str:
    """Strip emojis/quotes that gTTS would pronounce awkwardly; tidy spacing."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _QUOTE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", cleaned)
    return cleaned.strip()


def _extract_subject(caption: str) -> str:
    """Strip only a leading article, keep the caption's natural grammar."""
    words = re.findall(r"[A-Za-z']+", caption)
    if words and words[0].lower() in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words[:8]) or "a happy child"


def _is_safe_for_kids(text: str) -> bool:
    """Return False if ``text`` contains any word in the kid blocklist.

    Used to gate the output of the text-generation pipeline so any
    age-inappropriate word (most likely from the distilgpt2 fallback)
    is replaced by a vetted safe story instead.
    """
    if not text:
        return True
    words = set(re.findall(r"[A-Za-z']+", text.lower()))
    return not (words & _KID_BLOCKLIST)


def _safe_fallback_story(caption: str) -> str:
    """Return a pre-written, vetted safe story.

    Used when the model's output fails the kid-safety check. Subject is
    pulled from the caption when possible so the fallback still feels
    on-topic to the user.
    """
    subject = _extract_subject(caption) or "a happy child"
    return (
        f"Once upon a time, there was {subject} who loved to play. "
        f"They ran and jumped and laughed with all their friends. "
        f"They shared toys, snacks, and big warm hugs. "
        f"At the end of the day, they went home feeling tired and happy. "
        f"What a wonderful day it was! "
        f"They played together every day, sharing toys, snacks, and big "
        f"warm hugs, until the stars came out and they fell asleep smiling. "
        f"And they all lived happily ever after."
    )


def _truncate_to_word_count(text: str, *, low: int, high: int) -> str:
    """Trim to ≤ high words, always ending on a COMPLETE sentence.

    If the trimmed story falls under ``low`` words, pad with distinct
    closers (never the same line twice) so the final count lands inside
    the ``[low, high]`` window required by the assignment.
    """
    if not text:
        return text

    sentences = [s for s in re.split(
        r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return text

    kept: list[str] = []
    count = 0
    for s in sentences:
        n = len(s.split())
        if count + n > high and kept:
            break
        kept.append(s)
        count += n

    out = " ".join(kept).strip()

    # Pad toward the low bound with distinct closers, not the same line.
    for closer in _EXTRA_CLOSERS:
        if 0 < len(out.split()) < low:
            out = out.rstrip() + closer
        else:
            break
    return out


# Trailing filler words to strip when a sentence is cut mid-phrase.
_TRAILING_FILLER = re.compile(
    r"\b(a|an|the|and|but|or|so|because|with|for|to|of|in|on|at|"
    r"was|were|is|are|be|been|being)\s*$",
    re.IGNORECASE,
)
_DANGLING_LETTER = re.compile(r"\s+[A-Z]\.\s*$")


def _repair_story_end(text: str) -> str:
    """Fix ragged endings: dangling articles, trailing single letters,
    missing punctuation. Trims back to the last COMPLETE sentence.

    Only repairs the ending; padding to reach the word-count minimum is
    handled by ``_truncate_to_word_count``, and appending the canonical
    "happily ever after" closer is handled by ``_ensure_happy_ending``.
    Splitting those two steps means the happy ending is never dropped
    by the word-count cap.
    """
    if not text:
        return text

    out = _DANGLING_LETTER.sub("", text.strip()).rstrip()

    if out and not out.endswith((".", "!", "?")):
        last_punct = max(out.rfind(p) for p in ".!?")
        if last_punct > 0:
            out = out[: last_punct + 1].rstrip()

    # Loop so multi-word trailing fillers ("in the", "and was") strip
    # in one pass.
    while _TRAILING_FILLER.search(out):
        out = _TRAILING_FILLER.sub("", out).rstrip()

    return out


def _ensure_happy_ending(text: str) -> str:
    """Append "And they all lived happily ever after." if not already
    present. Called AFTER ``_truncate_to_word_count`` so the appended
    sentence is never dropped by the 100-word cap.
    """
    if not text:
        return text
    if not re.search(
        r"(happily ever after|the end)\s*[.!]?\s*$", text, re.IGNORECASE
    ):
        text = text.rstrip() + _HAPPY_ENDING
    return text


# ---------------------------------------------------------------------------
# 3b. Session-state helpers
# ---------------------------------------------------------------------------

# Sentinel marker used by _apply_session_state to distinguish "init if
# missing" from "overwrite". Cannot collide with any real default value
# because dict[str, Any] defaults are JSON-compatible scalars.
_SETDEFAULTS: Any = object()


def _apply_session_state(mode: Any) -> None:
    """Bulk-assign every ``DEFAULT_SESSION_STATE`` value to ``st.session_state``.

    Pass ``_SETDEFAULTS`` to call ``setdefault`` (init-only), or ``None``
    to overwrite every key back to its default.
    """
    for key, default in DEFAULT_SESSION_STATE.items():
        if mode is _SETDEFAULTS:
            st.session_state.setdefault(key, default)
        else:
            st.session_state[key] = default


def _init_session_state() -> None:
    """Populate st.session_state with default values on first run."""
    _apply_session_state(_SETDEFAULTS)


def _reset_session_state() -> None:
    """Reset every StorySpark session key back to its default value."""
    _apply_session_state(None)


# ---------------------------------------------------------------------------
# 3c. Pipeline error helpers — kid-friendly surfaces for failures
# ---------------------------------------------------------------------------

# Friendly messages keyed by failure stage. Centralising them here keeps
# the UI copy consistent and makes them easy to translate / unit-test
# later. Each one is short, kid-appropriate, and tells the user what
# to do next instead of exposing technical details.

_ERR_NO_IMAGE: str = (
    "👆 Please upload a picture first — then tap the magic button!"
)
_ERR_BAD_IMAGE: str = (
    "😯 We couldn't read that picture — it looks broken or "
    "isn't really a PNG / JPG / WEBP. Please try a different one!"
)
_ERR_OFFLINE: str = (
    "🌐 It looks like you're offline! "
    "Please check your internet connection and try again."
)
_ERR_MODEL_LOAD: str = (
    "🤖 Our story-building models are taking a little nap. "
    "Please try again in a moment!"
)
_ERR_CAPTION: str = (
    "🔍 We had trouble looking at your picture. "
    "Please try again in a moment!"
)
_ERR_EMPTY_CAPTION: str = (
    "🤔 We couldn't find anything to describe in this picture. "
    "Try a different image with a clearer subject!"
)
_ERR_STORY: str = (
    "✍️ We had trouble writing your story. Please tap the button again!"
)
_ERR_EMPTY_STORY: str = (
    "🤔 Our story-writer didn't come up with anything this time. "
    "Please tap the button again to try once more!"
)
_ERR_TTS_NETWORK: str = (
    "📖 We wrote your story! 🎤 …but couldn't record the voice — "
    "you appear to be offline. Reconnect and tap again to add audio!"
)
_ERR_TTS_OTHER: str = (
    "📖 We wrote your story! 🎤 …but the voice recording didn't work. "
    "Please try again in a moment to add audio!"
)


def _reset_pipeline_state() -> None:
    """Reset phase/progress/loader after a pipeline failure.

    Used by every stage-specific error handler so the user can retry
    without manual state cleanup. Safe to call when no loader exists.
    """
    slot = st.session_state.get("_loader_slot")
    if slot is not None:
        try:
            slot.empty()
        except Exception:                   # noqa: BLE001 — defensive
            logger.exception("Failed to clear loader slot")
    st.session_state.pop("_loader_slot", None)
    st.session_state.phase = "idle"
    st.session_state.progress_step = 0


def _validate_word_count(story: str) -> None:
    """Warn (don't block) if a story lands outside the 50–100 word target.

    The generator tries hard to land in this window — see
    ``_truncate_to_word_count`` and ``_ensure_happy_ending`` — but if
    a model output ever slips through, we surface a friendly warning
    instead of silently shipping an off-spec story.
    """
    if not story:
        return
    n = len(story.split())
    if n < STORY_WORD_COUNT_LOW or n > STORY_WORD_COUNT_HIGH:
        st.warning(
            f"📏 Your story is {n} words (the assignment target is "
            f"{STORY_WORD_COUNT_LOW}–{STORY_WORD_COUNT_HIGH} words). "
            "Tap the magic button again for a different story!"
        )


def _looks_offline(exc: BaseException) -> bool:
    """True if ``exc`` (or any chained cause) looks like a network failure.

    Lets the pipeline route a wider range of low-level errors into the
    friendly "you're offline" bucket instead of dumping them verbatim.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _NETWORK_ERRORS):
            return True
        # gTTS wraps network failures in gTTSError; the chained cause
        # is the real URLError. Walk the chain to find it.
        next_exc = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
        current = next_exc if isinstance(next_exc, BaseException) else None
    return False


# ---------------------------------------------------------------------------
# 4. UI — small HTML-fragment helpers
# ---------------------------------------------------------------------------

# Quest-path steps used by _render_quest_path(). Tuples are
# (icon, title, sub). The '&' in "Listen & save" is pre-escaped for HTML.
_QUEST_STEPS: list[tuple[str, str, str]] = [
    ("🎨", "Pick a picture",    "PNG · JPG · WEBP · up to 25 MB"),
    ("✨", "Make my story",     "Tap the big pink button"),
    ("🎧", "Listen &amp; save", "Hear your story come alive"),
]


def _render_floating_scene() -> None:
    """Render the fixed-position floating background emoji scene."""
    _html(
        '<div class="scene">'
        '<span class="float f1">🌈</span><span class="float f2">⭐</span>'
        '<span class="float f3">🎈</span><span class="float f4">☁️</span>'
        '<span class="float f5">✨</span><span class="float f6">🦄</span>'
        '</div>'
    )


def _render_title() -> None:
    """Render the rainbow 'StorySpark' title and tagline."""
    _html(
        '<div class="title">'
        '<div class="title-main">✨ StorySpark ✨</div>'
        '<div class="title-sub">Show me a picture and I\'ll tell you a story!</div>'
        '</div>'
    )


def _render_quest_path(active_step: int) -> None:
    """Render the 3-step 'Pick → Make → Listen' quest path (empty state)."""
    items: list[str] = []
    for n, (icon, title, sub) in enumerate(_QUEST_STEPS, start=1):
        if n < active_step:
            cls, badge = "quest-step done", "✓"
        elif n == active_step:
            cls, badge = "quest-step active", str(n)
        else:
            cls, badge = "quest-step", str(n)
        items.append(
            f'<div class="{cls}">'
            f'<div class="quest-badge">{badge}</div>'
            f'<div class="quest-icon">{icon}</div>'
            f'<div class="quest-text">'
            f'<div class="quest-title">{title}</div>'
            f'<div class="quest-sub">{sub}</div>'
            f'</div></div>'
        )
        if n < len(_QUEST_STEPS):
            items.append('<div class="quest-connector"></div>')
    _html(f'<div class="quest-path">{"".join(items)}</div>')


def _render_progress_loader(step: int) -> None:
    """Render the inline turning-book loader with progress bar and emoji."""
    emoji, pct, note = LOADER_STEPS[min(step, len(LOADER_STEPS) - 1)]
    _html(
        f'<div class="progress-inline">'
        f'<div class="turning-book" aria-hidden="true">'
        f'<div class="base"><div class="left"></div><div class="right"></div></div>'
        f'<div class="spine"></div>'
        f'<div class="page">'
        f'<span class="line l1"></span><span class="line l2"></span>'
        f'<span class="line l3"></span><span class="line l4"></span>'
        f'</div></div>'
        f'<div class="progress-emoji">{emoji}</div>'
        f'<div class="progress-bar">'
        f'<div class="progress-fill" style="width:{pct*100:.0f}%"></div>'
        f'</div>'
        f'<div class="progress-text">{note}</div>'
        f'</div>'
    )


# (label, filename, MIME prefix) for the two download links. The icon
# SVG is hard-coded in style.css, so it's not part of this table.
_DOWNLOAD_LINKS: list[tuple[str, str, str]] = [
    ("Save the voice", "storyspark_story.mp3", "data:audio/mp3;base64,"),
    ("Save the story", "storyspark_story.txt", "data:text/plain;charset=utf-8,"),
]


def _render_download_buttons(audio_bytes: bytes, story: str) -> None:
    """Render two raw ``<a download>`` links (voice + story).

    Raw links are used instead of ``st.download_button`` so clicking
    them does NOT trigger a Streamlit rerun and interrupt audio playback.
    """
    payloads = [
        base64.b64encode(audio_bytes).decode("ascii") if audio_bytes else "",
        quote(story, safe=""),
    ]
    css_class = ["ss-dl-voice", "ss-dl-story"]
    cols = st.columns(len(_DOWNLOAD_LINKS), gap="small")
    for col, (label, filename, mime), payload, css in zip(
        cols, _DOWNLOAD_LINKS, payloads, css_class
    ):
        with col:
            _html(
                f'<div class="ss-dl-cell">'
                f'<a class="ss-dl-link {css}" '
                f'href="{mime}{payload}" download="{filename}">'
                f'{label}</a></div>'
            )


def _render_story_output(story: str, audio_bytes: bytes) -> None:
    """Render the completed story: heading, text, word count, audio, downloads."""
    if not st.session_state.celebrated:
        st.balloons()
        st.session_state.celebrated = True

    _page_heading("📖", "Here comes your story!")
    _html(f'<div class="story-text">{story}</div>')

    word_count = len(story.split())
    _html(
        f'<div class="word-count">'
        f'<span class="wc-emoji">📝</span>'
        f'<span class="wc-num">{word_count}</span>'
        f'<span class="wc-sep">words</span>'
        f'</div>'
    )

    st.audio(audio_bytes, format="audio/mp3")
    _render_download_buttons(audio_bytes, story)


def _render_empty_state() -> None:
    """Render Ollie the Story Owl plus the original guidance text."""
    _html(
        '<div class="page-empty">'
        # Ollie the Story Owl
        '<div class="story-guide" aria-hidden="true">'
        '<div class="guide-body">'
        '<div class="guide-eyes">'
        '<span class="eye"><span class="pupil"></span></span>'
        '<span class="eye"><span class="pupil"></span></span>'
        '</div>'
        '<div class="guide-beak"></div>'
        '</div>'
        '<div class="guide-wing left"></div>'
        '<div class="guide-wing right"></div>'
        '</div>'
        # Speech bubble
        '<div class="guide-bubble">'
        '<span class="guide-greet">Hi, story-maker!</span>'
        '<span class="guide-ask">'
        'I\'m Ollie the Story Owl 🦉<br>'
        'Show me a picture and I\'ll<br>'
        'tell you a tale just for you…'
        '</span>'
        '</div>'
        # Original guidance text
        '<div class="empty-original">'
        '<div class="empty-text">Your story will appear here…</div>'
        '<div class="empty-hint">'
        '🖼️ Upload a picture on the left → 🎨 press the magic button!'
        '</div>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# 5. UI — page-level renderers
# ---------------------------------------------------------------------------

def render_upload_page() -> tuple[Any, bool]:
    """Render the LEFT (upload) page.

    Returns
    -------
    uploaded : Any
        The currently uploaded file (``UploadedFile``), or ``None``.
    make_story : bool
        True only on the run where the user just clicked 'Make My Story!'.
    """
    _page_heading("🎨", "Put your picture here!")

    uploaded = st.file_uploader(
        label="Upload a picture",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="uploaded_image",
    )

    if uploaded is None:
        # Empty state: show the quest path. Stale-state reset in main().
        active_step = PHASE_TO_ACTIVE_STEP.get(st.session_state.phase, 1)
        _render_quest_path(active_step)
        return uploaded, False

    # Rewind the buffer so a second Image.open() in the pipeline sees
    # the same bytes.
    uploaded.seek(0)

    # Validate the upload RIGHT HERE so corrupt / truncated files fail
    # with a friendly message instead of blowing up deep inside the
    # pipeline. PIL's ``Image.open()`` is lazy — it only parses the
    # header — so a file with a valid header but a truncated body
    # (e.g. an interrupted download) would otherwise look fine here
    # and only crash when the BLIP pipeline tries to read pixels.
    # Calling ``.load()`` forces a full decode NOW, catching both
    # "not an image at all" (``UnidentifiedImageError``) and "valid
    # header, body cut off" (``OSError`` on decode).
    try:
        preview_image = Image.open(uploaded)
        preview_image.load()
    except (Image.UnidentifiedImageError, OSError) as exc:
        logger.warning("User uploaded an unreadable image: %s", exc)
        st.error(_ERR_BAD_IMAGE)
        # Skip the preview + button so the user can't trigger the
        # pipeline on a known-bad file.
        return uploaded, False

    st.image(preview_image, use_container_width=True)

    if st.session_state.phase == "working":
        _html(
            '<style>[data-testid="stFileUploader"]'
            '{pointer-events:none;opacity:0.55;}</style>'
        )

    # Center the button below the uploaded image using a 3-column layout.
    # The center column is wider so the button has room to breathe.
    _left, _center, _right = st.columns([1, 4, 1])
    with _center:
        make_story = st.button(
            "Make My Story!",
            type="primary",
            help="Tap to make a story from your picture ✨",
            key="make_story_btn",
            disabled=(st.session_state.phase == "working"),
        )

    # Happy-path return: the file is valid and the button is on screen.
    # Returning (uploaded, make_story) lets the caller in main() unpack
    # the result and forward the button-click into _run_story_pipeline.
    # Without this, the function would fall off the end and implicitly
    # return None, which would crash the `uploaded, make_story = ...`
    # unpacking in main().
    return uploaded, make_story


def render_story_page() -> None:
    """Render the RIGHT (story) page: loader, finished story, or empty state."""
    story = st.session_state.story
    phase = st.session_state.phase

    if phase == "working":
        # Create the empty slot ONCE here, then the pipeline paints into it
        # in-place. Storing it in session_state avoids creating a second slot
        # (which would show two loaders at once).
        if "_loader_slot" not in st.session_state:
            st.session_state._loader_slot = st.empty()
        with st.session_state._loader_slot.container():
            _render_progress_loader(st.session_state.progress_step)
    elif story:
        # Clear any leftover loader slot from the previous phase so it
        # doesn't linger under the finished story.
        if "_loader_slot" in st.session_state:
            del st.session_state._loader_slot
        _render_story_output(story, st.session_state.audio_bytes)
    else:
        _render_empty_state()


# ---------------------------------------------------------------------------
# 6. Main pipeline — two-phase state machine
# ---------------------------------------------------------------------------

def _clear_stale_state_if_needed(uploaded: Any) -> None:
    """If the user removed their upload but a story still exists, reset state.

    Kept outside render functions so render stays side-effect free.
    """
    if uploaded is None and (st.session_state.story or st.session_state.audio_bytes):
        _reset_session_state()
        st.rerun()


def _run_story_pipeline(uploaded: Any, make_story: bool) -> None:
    """Drive the image → story → audio state machine.

    Runs in two phases via Streamlit reruns:

    * Phase 1 — user clicks 'Make My Story!': flip phase to 'working'
      and rerun, so the loader UI paints immediately.
    * Phase 2 — phase is 'working': execute the pipeline, painting the
      loader in-place between stages, then flip to 'done' (or back to
      'idle' on failure).

    Each pipeline stage has its own try/except so the user sees a
    kid-friendly message that names *what* failed and *what to do*,
    instead of a single generic error that exposes raw tracebacks.
    Network failures get a dedicated branch; TTS failures preserve
    the already-generated story so the user can still read it.
    """
    # ---- Phase 1: user just clicked the magic button ----
    if (
        make_story
        and uploaded is not None
        and st.session_state.phase == "idle"
    ):
        st.session_state.phase = "working"
        st.session_state.progress_step = 0
        st.rerun()

    # Defensive guard: if a click ever arrives without a file (shouldn't
    # happen because the button is hidden in the empty state, but a
    # stale session_state could let it slip through), tell the user
    # what's missing instead of silently returning.
    if make_story and uploaded is None and st.session_state.phase == "idle":
        st.warning(_ERR_NO_IMAGE)
        return

    # ---- Phase 2: actually run the pipeline ----
    if st.session_state.phase != "working" or uploaded is None:
        return

    uploaded.seek(0)
    try:
        working_image = Image.open(uploaded)
        working_image.load()             # force full decode now
    except (Image.UnidentifiedImageError, OSError) as exc:
        # Defensive: render_upload_page already validated this file,
        # but a stale session_state could let an invalid upload slip
        # through. Surface the same friendly message either way.
        logger.warning("Pipeline image re-validation failed: %s", exc)
        _reset_pipeline_state()
        st.error(_ERR_BAD_IMAGE)
        return

    # Initialise BEFORE the per-stage try blocks so every binding
    # exists even on early-return.
    caption, story, audio_bytes = "", "", b""

    # Reuse the loader slot created by render_story_page so we don't get
    # two .progress-inline blocks on screen at once. If it's missing
    # (e.g. pipeline triggered outside the right column), create one.
    loader_slot = st.session_state.get("_loader_slot") or st.empty()
    st.session_state._loader_slot = loader_slot

    def _paint(step_index: int) -> None:
        with loader_slot.container():
            _render_progress_loader(step_index)

    # ----------------------------------------------------------------
    # Stage 1: image captioning
    # ----------------------------------------------------------------
    try:
        _paint(0)
        caption = img2text(working_image)
    except _NETWORK_ERRORS as exc:
        # Image was fetched (or supplied) but the model call needs HF.
        logger.warning("Network error during captioning: %s", exc)
        _reset_pipeline_state()
        st.error(_ERR_OFFLINE)
        return
    except Exception:                                   # noqa: BLE001
        logger.exception("Captioning failed")
        _reset_pipeline_state()
        st.error(_ERR_CAPTION)
        return

    if not caption.strip():
        logger.info("Captioner returned an empty string")
        _reset_pipeline_state()
        st.warning(_ERR_EMPTY_CAPTION)
        return

    # ----------------------------------------------------------------
    # Stage 2: story generation
    # ----------------------------------------------------------------
    try:
        _paint(1)
        story = text2story(caption)
    except _NETWORK_ERRORS as exc:
        logger.warning("Network error during story generation: %s", exc)
        _reset_pipeline_state()
        st.error(_ERR_OFFLINE)
        return
    except RuntimeError as exc:
        # _get_story_pipeline raises RuntimeError when BOTH models
        # fail to load — surface it as "models are napping" instead of
        # leaking the underlying HF / OS error to the user.
        logger.error("Story model unavailable: %s", exc)
        _reset_pipeline_state()
        st.error(_ERR_MODEL_LOAD)
        return
    except Exception:                                   # noqa: BLE001
        logger.exception("Story generation failed")
        _reset_pipeline_state()
        st.error(_ERR_STORY)
        return

    if not story.strip():
        logger.info("text2story returned an empty string")
        _reset_pipeline_state()
        st.warning(_ERR_EMPTY_STORY)
        return

    _paint(2)
    # Surface (but don't block on) word-count drift. Generator tries
    # hard to land in [50, 100]; this is the safety net.
    _validate_word_count(story)

    # ----------------------------------------------------------------
    # Stage 3: text-to-speech
    # ----------------------------------------------------------------
    # TTS only needs an internet call (gTTS → Google). If it fails we
    # still save the story so the user can read it; only audio is lost.
    try:
        audio_bytes = text2audio(story)
    except _NETWORK_ERRORS as exc:
        logger.warning("Network error during TTS: %s", exc)
        st.session_state.story = story
        st.session_state.audio_bytes = b""
        st.session_state.caption = caption
        st.session_state.celebrated = False
        st.session_state.phase = "done"
        st.session_state.progress_step = 0
        st.session_state._loader_slot = None
        st.error(_ERR_TTS_NETWORK)
        st.rerun()
        return
    except Exception as exc:                            # noqa: BLE001
        # gTTSError (and any other TTS-layer failure) goes here. We
        # also check _looks_offline() so a wrapped URLError inside a
        # gTTSError gets the friendlier offline message.
        if _looks_offline(exc):
            logger.warning("TTS failed with underlying network error: %s", exc)
            user_msg = _ERR_TTS_NETWORK
        else:
            logger.exception("TTS failed")
            user_msg = _ERR_TTS_OTHER
        st.session_state.story = story
        st.session_state.audio_bytes = b""
        st.session_state.caption = caption
        st.session_state.celebrated = False
        st.session_state.phase = "done"
        st.session_state.progress_step = 0
        st.session_state._loader_slot = None
        st.error(user_msg)
        st.rerun()
        return

    # ----------------------------------------------------------------
    # Stage 4: final paint + commit
    # ----------------------------------------------------------------
    _paint(3)
    time.sleep(0.25)                  # let the final frame paint

    st.session_state.story = story
    st.session_state.audio_bytes = audio_bytes
    st.session_state.caption = caption
    st.session_state.celebrated = False
    st.session_state.phase = "done"
    st.session_state.progress_step = 0
    st.session_state._loader_slot = None
    st.rerun()


# ---------------------------------------------------------------------------
# 7. Footer
# ---------------------------------------------------------------------------

def _render_footer() -> None:
    """Render the fixed-position footer bar at the bottom of the page."""
    _html("<div class='footer-bar'>Made with 💖 for tiny story-lovers</div>")


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full StorySpark Streamlit application."""
    _init_session_state()
    _render_floating_scene()
    _render_title()

    page_l, page_r = st.columns([1, 1], gap="large")
    with page_l:
        uploaded, make_story = render_upload_page()
    with page_r:
        render_story_page()

    _clear_stale_state_if_needed(uploaded)
    _run_story_pipeline(uploaded, make_story)
    _render_footer()


main()
