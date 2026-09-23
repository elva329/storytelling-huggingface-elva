# =============================================================================
# StorySpark — A Storytelling Application for Kids (ages 3–10)
# =============================================================================
#
#  Course : ISOM5240 — Deep Learning Business Applications with Python
#  Author : Elva
#  Stack  : Streamlit · Hugging Face Transformers · gTTS
#
#  Pipeline
#  --------
#    [Image upload]
#        │  1. img2text()  →  pipeline("image-to-text")
#        │                   (Salesforce/blip-image-captioning-base)
#        ▼
#    2. text2story()  →  pipeline("text-generation")
#                        (roneneldan/TinyStories-33M, fallback: distilgpt2)
#        │
#        ▼
#    3. text2audio()  →  gTTS (Google Text-to-Speech)
#        │
#        ▼
#    [Storybook spread with audio player]
#
#  Product spec
#  ------------
#  • Stories must be 50–100 words. The generator trims to the upper
#    bound; _validate_word_count() warns if a model output slips out.
#  • Every word the user reads must come from the LLM. No hard-coded
#    sentences are ever appended.
#  • The primary "Make My Story!" button must not reflow when its
#    enabled/disabled state toggles. See components.css for the
#    CSS-only lockout rationale.
#
#  All styling lives in ./styles/.
# =============================================================================

from __future__ import annotations

import base64
import io
import logging
import re
import time
import urllib.error
from enum import Enum
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

from constants import (
    _EMOJI_RE,
    _KID_BLOCKLIST,
    _QUOTE_RE,
    _TRAILING_FILLER,
    _DANGLING_LETTER,
    _LEADING_ARTICLES,
)

# gTTSError moved between gTTS modules across releases; import defensively.
try:
    from gtts import gTTSError                     # type: ignore[attr-defined]
except ImportError:                                # pragma: no cover
    from gtts.tts import gTTSError                 # type: ignore[attr-defined]

# Exceptions that indicate the user's network is unavailable. Grouped
# into a single "you're offline" bucket in the UI.
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
    """Inject the project's stylesheets in load order.

    Later files override earlier ones on selector ties, so the chain
    MUST be: style.css → components.css → animations.css → responsive.css.

    Paths are resolved relative to this file's directory so the app
    works on Streamlit Cloud, where cwd may differ from source dir.
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
    """Render raw HTML via Streamlit."""
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

# How long to hold the "All done!" loader frame before committing the
# story and rerunning. Small value: the loader is a UX nicety, not a
# requirement, so we don't want to block the user for long.
_FINAL_FRAME_DELAY_S: float = 0.25

_STORY_PROMPTS: dict[str, str] = {
    KID_STORY_MODEL: "Once upon a time there was a {subject}. ",
    FALLBACK_STORY_MODEL: (
        "Tell a happy bedtime story for a 6 year old child about {subject}. "
        "Use simple words, short sentences, and end with something nice. "
        "Story:\n"
    ),
}
_FALLBACK_PROMPT: str = _STORY_PROMPTS[FALLBACK_STORY_MODEL]

PHASE_TO_ACTIVE_STEP: dict[str, int] = {"idle": 1, "working": 2, "done": 3}

DEFAULT_SESSION_STATE: dict[str, Any] = {
    "story": "",
    "audio_bytes": b"",
    "caption": "",
    "celebrated": False,
    "phase": "idle",           # idle | working | done
    "progress_step": 0,
    "_error": "",              # deferred message; re-emitted next render
}

STORY_WORD_COUNT_LOW: int = 50
STORY_WORD_COUNT_HIGH: int = 100
# Alias kept separate so a future change to the trim target does not
# silently change the validation window (or vice versa).
STORY_WORD_COUNT_TRIM_HIGH: int = STORY_WORD_COUNT_HIGH

# (emoji, fractional progress, status text). Index matches
# st.session_state.progress_step (0..3).
LOADER_STEPS: list[tuple[str, float, str]] = [
    ("🔎", 0.15, "Looking at your picture"),
    ("📖", 0.45, "Writing your story"),
    ("🎤", 0.80, "Recording the voice"),
    ("✅", 1.00, "All done!"),
]


class UploadState(str, Enum):
    """Result of render_upload_page(). Callers branch on this rather
    than on a `make_story` boolean, so 'no file' and 'bad file' are
    distinguishable without relying on session_state."""
    NO_FILE = "no_file"
    BAD_FILE = "bad_file"
    READY = "ready"


# ---------------------------------------------------------------------------
# 2. Cached Hugging Face pipelines
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_caption_pipeline() -> Any:
    """Load the BLIP image-to-text pipeline (cached across sessions).

    Tries the high-level ``pipeline()`` first. On failure (usually a
    missing torchvision / sentencepiece in the host environment), falls
    back to loading the processor and model directly and wrapping them
    in a small callable that mimics the pipeline output shape.

    ``transformers.pipeline`` has ~30 task-specific overloads; the
    ``"image-to-text"`` task is not fully covered by the type stubs, so
    the ``task=`` argument carries an explicit type-ignore.
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
        """Drop-in for ``pipeline('image-to-text')``.

        Explicit ``__init__`` binding keeps Pylance happy: passing the
        processor and model as constructor arguments means they're
        typed as ``Any`` rather than unresolved closure variables, so
        the ``__call__`` body type-checks cleanly.
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
def _get_story_pipeline() -> tuple[Any, str, int | None]:
    """Load TinyStories (preferred) with distilgpt2 as a fallback.

    Returns ``(pipeline, model_id, pad_token_id)``. ``pad_token_id`` is
    cached here so callers don't have to re-inspect the tokenizer on
    every generation. Raises ``RuntimeError`` if no model loads.
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
            pad_id = getattr(tokenizer, "pad_token_id", None)
            return text_gen, model_id, pad_id
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not load %s: %s", model_id, exc)
            continue
    raise RuntimeError("No story-generation model could be loaded.")


# ---------------------------------------------------------------------------
# 3. Core pipeline: image → caption → story → audio
# ---------------------------------------------------------------------------

def img2text(image: Image.Image) -> str:
    """Caption an image with the BLIP image-to-text pipeline.

    Returns a capitalised, stripped caption, or an empty string on
    decode failure. Only PIL Images are accepted; the app never
    fetches remote URLs.
    """
    captioner = _get_caption_pipeline()

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    results = captioner(rgb, max_new_tokens=40)
    text = (results[0].get("generated_text", "") if results else "").strip()
    return text[0].upper() + text[1:] if text else text


def text2story(caption: str) -> str:
    """Generate a kid-friendly bedtime story (50–100 words) from a caption.

    Emojis and stray quotes are stripped, the ending is repaired to the
    last complete sentence, and the result is trimmed to the product
    spec's word-count window. Every word comes from the model — no
    hard-coded sentences are appended.
    """
    story_gen, model_id, pad_id = _get_story_pipeline()
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
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    outputs = story_gen(prompt, **gen_kwargs)
    generated = str(outputs[0].get("generated_text", "")) if outputs else ""

    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    generated = re.sub(r"\s+", " ", generated).strip()
    generated = _strip_emojis(generated)
    generated = _repair_story_end(generated)
    generated = _truncate_to_word_count(
        generated, high=STORY_WORD_COUNT_TRIM_HIGH)
    if not _is_safe_for_kids(generated):
        logger.warning(
            "Story output failed the kid-safety check; shipping the "
            "model output anyway per product spec."
        )
    return _strip_emojis(generated)


def text2audio(story_text: str) -> bytes:
    """Convert a story to MP3 bytes via Google Text-to-Speech.

    Emojis and quotes are stripped first so gTTS does not attempt to
    pronounce them. Returns empty bytes for empty input.
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

def _strip_emojis(text: str) -> str:
    """Remove emojis and quotes that gTTS would mispronounce; tidy spacing."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = _QUOTE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", cleaned)
    return cleaned.strip()


def _extract_subject(caption: str) -> str:
    """Strip only a leading article; keep the caption's natural grammar."""
    words = re.findall(r"[A-Za-z']+", caption)
    if words and words[0].lower() in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words[:8]) or "a happy child"


def _is_safe_for_kids(text: str) -> bool:
    """True unless ``text`` contains any word in the kid blocklist.

    Used to log a warning when the text-generation pipeline emits an
    age-inappropriate word (most likely from the distilgpt2 fallback).
    The story is still shipped per the product spec.
    """
    if not text:
        return True
    words = set(re.findall(r"[A-Za-z']+", text.lower()))
    return not (words & _KID_BLOCKLIST)


def _truncate_to_word_count(text: str, *, high: int) -> str:
    """Trim to ≤ ``high`` words, always ending on a complete sentence.

    Returns the input unchanged if it is already within the limit or
    contains no sentence boundaries. Off-spec short stories are
    surfaced by ``_validate_word_count`` after this function returns.
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

    return " ".join(kept).strip()


def _repair_story_end(text: str) -> str:
    """Fix ragged endings: dangling articles, trailing single letters,
    missing punctuation. Trims back to the last complete sentence.

    Only touches the ending; the word-count window is enforced later
    by ``_truncate_to_word_count``.
    """
    if not text:
        return text

    out = _DANGLING_LETTER.sub("", text.strip()).rstrip()

    if out and not out.endswith((".", "!", "?")):
        last_punct = max(out.rfind(p) for p in ".!?")
        if last_punct > 0:
            out = out[: last_punct + 1].rstrip()

    # Loop so multi-word fillers ("in the", "and was") strip in one pass.
    while _TRAILING_FILLER.search(out):
        out = _TRAILING_FILLER.sub("", out).rstrip()

    return out


# ---------------------------------------------------------------------------
# 3b. Session-state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Populate st.session_state with default values on first run."""
    for key, default in DEFAULT_SESSION_STATE.items():
        st.session_state.setdefault(key, default)


def _reset_session_state() -> None:
    """Reset every StorySpark session key back to its default value."""
    for key, default in DEFAULT_SESSION_STATE.items():
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# 3c. Pipeline error helpers — kid-friendly surfaces for failures
# ---------------------------------------------------------------------------

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


def _fail(
    message: str,
    *,
    keep_story: str = "",
    keep_caption: str = "",
) -> None:
    """Surface a kid-friendly error and return the state machine to a
    safe state in a single call.

    Set ``keep_story`` to preserve a partially-successful output (used
    by the TTS stage, which loses only audio on failure). Otherwise the
    phase drops back to idle so the user can retry from the top.
    """
    slot = st.session_state.get("_loader_slot")
    if slot is not None:
        try:
            slot.empty()
        except Exception:                   # noqa: BLE001 — defensive
            logger.exception("Failed to clear loader slot")
    st.session_state.pop("_loader_slot", None)
    st.session_state.progress_step = 0

    if keep_story:
        st.session_state.story = keep_story
        st.session_state.audio_bytes = b""
        st.session_state.caption = keep_caption
        st.session_state.celebrated = False
        st.session_state.phase = "done"
    else:
        st.session_state.phase = "idle"

    # Defer the toast: st.rerun() clears it before the user reads it,
    # so we stash the text and re-emit it on the next render pass.
    st.session_state._error = message
    st.rerun()


def _flush_deferred_error() -> None:
    """Re-emit any error stashed by _fail() before the previous rerun."""
    message = st.session_state.pop("_error", "")
    if message:
        st.error(message)


def _validate_word_count(story: str) -> None:
    """Warn (without blocking) if a story lands outside the 50–100 word target.

    The generator tries to land in this window; this is the safety net
    that surfaces an off-spec output instead of silently shipping it.
    """
    if not story:
        return
    n = len(story.split())
    if n < STORY_WORD_COUNT_LOW or n > STORY_WORD_COUNT_HIGH:
        st.warning(
            f"📏 Your story is {n} words (the target is "
            f"{STORY_WORD_COUNT_LOW}–{STORY_WORD_COUNT_HIGH} words). "
            "Tap the magic button again for a different story!"
        )


def _looks_offline(exc: BaseException) -> bool:
    """True if ``exc`` or any chained cause looks like a network failure.

    gTTS wraps network failures in gTTSError, so the interesting cause
    is often one or two links down the ``__cause__`` / ``__context__``
    chain. Python exception chains cannot cycle, so the walk terminates.
    """
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, _NETWORK_ERRORS):
            return True
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None
    return False


# ---------------------------------------------------------------------------
# 4. UI — small HTML-fragment helpers
# ---------------------------------------------------------------------------

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


# (label, filename, MIME prefix). The icon SVG is hard-coded in
# components.css and is not part of this table.
_DOWNLOAD_LINKS: list[tuple[str, str, str]] = [
    ("Save the voice", "storyspark_story.mp3", "data:audio/mp3;base64,"),
    ("Save the story", "storyspark_story.txt", "data:text/plain;charset=utf-8,"),
]


def _render_download_buttons(audio_bytes: bytes, story: str) -> None:
    """Render two raw ``<a download>`` links (voice + story).

    Raw links are used instead of ``st.download_button`` so a click
    does not trigger a Streamlit rerun and interrupt audio playback.
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
        '<div class="guide-bubble">'
        '<span class="guide-greet">Hi, story-maker!</span>'
        '<span class="guide-ask">'
        'I\'m Ollie the Story Owl 🦉<br>'
        'Show me a picture and I\'ll<br>'
        'tell you a tale just for you…'
        '</span>'
        '</div>'
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

def render_upload_page() -> tuple[Any, UploadState, bool]:
    """Render the LEFT (upload) page.

    Returns ``(uploaded, state, make_story)`` where ``state`` is one of
    ``NO_FILE``, ``BAD_FILE`` or ``READY``, and ``make_story`` is True
    only on the run where the user just clicked 'Make My Story!'.
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
        active_step = PHASE_TO_ACTIVE_STEP.get(st.session_state.phase, 1)
        _render_quest_path(active_step)
        return uploaded, UploadState.NO_FILE, False

    # Rewind so a second Image.open() in the pipeline sees the same bytes.
    uploaded.seek(0)

    # Validate the upload here so corrupt / truncated files fail with a
    # friendly message instead of crashing deep inside the pipeline.
    # PIL's Image.open() is lazy (header only), so a file with a valid
    # header but a truncated body would otherwise look fine here and
    # blow up only when BLIP tries to read pixels. Image.load() forces
    # a full decode now, catching both UnidentifiedImageError and
    # "valid header, cut-off body" OSError.
    try:
        preview_image = Image.open(uploaded)
        preview_image.load()
    except (Image.UnidentifiedImageError, OSError) as exc:
        logger.warning("User uploaded an unreadable image: %s", exc)
        st.error(_ERR_BAD_IMAGE)
        return uploaded, UploadState.BAD_FILE, False

    st.image(preview_image, use_container_width=True)

    # The button is locked out during the working phase by CSS keyed on
    # the `.progress-inline` loader (see components.css). We deliberately
    # do NOT use Streamlit's `disabled=True`: that swaps the internal
    # DOM node and shifts the button 1-2px. The pipeline guard below
    # remains as defense-in-depth for any click that slips through.
    _left, _center, _right = st.columns([1, 4, 1])
    with _center:
        make_story = st.button(
            "Make My Story!",
            type="primary",
            key="make_story_btn",
        )

    return uploaded, UploadState.READY, make_story


def render_story_page() -> None:
    """Render the RIGHT (story) page: loader, finished story, or empty state."""
    story = st.session_state.story
    phase = st.session_state.phase

    if phase == "working":
        # Create the empty slot ONCE, then the pipeline paints into it
        # in place. Storing it in session_state prevents a second slot
        # from being created (which would show two loaders at once).
        if "_loader_slot" not in st.session_state:
            st.session_state._loader_slot = st.empty()
        with st.session_state._loader_slot.container():
            _render_progress_loader(st.session_state.progress_step)
    elif story:
        # Clear any leftover loader slot from the previous phase.
        if "_loader_slot" in st.session_state:
            del st.session_state._loader_slot
        _render_story_output(story, st.session_state.audio_bytes)
    else:
        _render_empty_state()


# ---------------------------------------------------------------------------
# 6. Main pipeline — two-phase state machine
# ---------------------------------------------------------------------------

def _clear_stale_state_if_needed(uploaded: Any) -> None:
    """Reset session state if the user removed their upload but a story
    still exists. The subsequent st.rerun() is a one-shot: the reset
    guarantees the condition is false on the next pass, so the rerun
    fires at most once per user-cleared upload."""
    if uploaded is None and (st.session_state.story or st.session_state.audio_bytes):
        _reset_session_state()
        st.rerun()


def _run_story_pipeline(
    uploaded: Any,
    state: UploadState,
    make_story: bool,
) -> None:
    """Drive the image → story → audio state machine.

    Runs in two Streamlit reruns:

    Phase 1 — user clicks 'Make My Story!': flip phase to 'working' and
    rerun so the loader paints immediately.

    Phase 2 — phase is 'working': execute each pipeline stage, painting
    the loader in place between stages, then flip to 'done' (or back to
    'idle' on failure).

    Each stage has its own try/except so the user sees a message that
    names *what* failed and *what to do*, rather than one generic error.
    Network failures get a dedicated branch; TTS failures preserve the
    already-generated story so the user can still read it.
    """
    # ---- Phase 1: user just clicked the magic button ----
    if (
        make_story
        and state is UploadState.READY
        and uploaded is not None
        and st.session_state.phase == "idle"
    ):
        st.session_state.phase = "working"
        st.session_state.progress_step = 0
        st.rerun()

    # Defensive guard: a click can only reach here without a valid file
    # if session_state is stale. Tell the user instead of silently
    # returning.
    if make_story and state is UploadState.NO_FILE and st.session_state.phase == "idle":
        st.warning(_ERR_NO_IMAGE)
        return

    # ---- Phase 2: actually run the pipeline ----
    if st.session_state.phase != "working" or uploaded is None:
        return

    uploaded.seek(0)
    try:
        working_image = Image.open(uploaded)
        working_image.load()
    except (Image.UnidentifiedImageError, OSError) as exc:
        # render_upload_page already validated this; a stale session
        # could still let an invalid upload slip through. Same message.
        logger.warning("Pipeline image re-validation failed: %s", exc)
        _fail(_ERR_BAD_IMAGE)
        return

    # Bind before the try blocks so every name exists on early return.
    caption, story, audio_bytes = "", "", b""

    # Reuse the loader slot from render_story_page so we don't render
    # two .progress-inline blocks at once. Create one if it's missing.
    loader_slot = st.session_state.get("_loader_slot") or st.empty()
    st.session_state._loader_slot = loader_slot

    def _paint(step_index: int) -> None:
        with loader_slot.container():
            _render_progress_loader(step_index)

    # ----- Stage 1: image captioning -----
    try:
        _paint(0)
        caption = img2text(working_image)
    except _NETWORK_ERRORS as exc:
        logger.warning("Network error during captioning: %s", exc)
        _fail(_ERR_OFFLINE)
        return
    except Exception:                                   # noqa: BLE001
        logger.exception("Captioning failed")
        _fail(_ERR_CAPTION)
        return

    if not caption.strip():
        logger.info("Captioner returned an empty string")
        _fail(_ERR_EMPTY_CAPTION)
        return

    # ----- Stage 2: story generation -----
    try:
        _paint(1)
        story = text2story(caption)
    except _NETWORK_ERRORS as exc:
        logger.warning("Network error during story generation: %s", exc)
        _fail(_ERR_OFFLINE)
        return
    except RuntimeError as exc:
        # _get_story_pipeline raises RuntimeError when both models fail
        # to load. Surface as "models are napping" instead of leaking
        # the underlying HF / OS error.
        logger.error("Story model unavailable: %s", exc)
        _fail(_ERR_MODEL_LOAD)
        return
    except Exception:                                   # noqa: BLE001
        logger.exception("Story generation failed")
        _fail(_ERR_STORY)
        return

    if not story.strip():
        logger.info("text2story returned an empty string")
        _fail(_ERR_EMPTY_STORY)
        return

    _paint(2)
    _validate_word_count(story)

    # ----- Stage 3: text-to-speech -----
    # TTS only needs an internet call (gTTS → Google). If it fails we
    # still save the story so the user can read it; only audio is lost.
    try:
        audio_bytes = text2audio(story)
    except _NETWORK_ERRORS as exc:
        logger.warning("Network error during TTS: %s", exc)
        _fail(_ERR_TTS_NETWORK, keep_story=story, keep_caption=caption)
        return
    except Exception as exc:                            # noqa: BLE001
        if _looks_offline(exc):
            logger.warning("TTS failed with underlying network error: %s", exc)
            _fail(_ERR_TTS_NETWORK, keep_story=story, keep_caption=caption)
        else:
            logger.exception("TTS failed")
            _fail(_ERR_TTS_OTHER, keep_story=story, keep_caption=caption)
        return

    # ----- Stage 4: commit + final paint -----
    # The story is committed BEFORE the "All done!" frame is painted so
    # an interrupt between the two never leaves the user looking at a
    # finished loader with no story.
    st.session_state.story = story
    st.session_state.audio_bytes = audio_bytes
    st.session_state.caption = caption
    st.session_state.celebrated = False
    st.session_state.phase = "done"
    st.session_state.progress_step = 0
    st.session_state._loader_slot = None

    _paint(3)
    time.sleep(_FINAL_FRAME_DELAY_S)
    st.rerun()


# ---------------------------------------------------------------------------
# 7. Footer
# ---------------------------------------------------------------------------

def _render_footer() -> None:
    """Render the fixed-position footer bar."""
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
        uploaded, state, make_story = render_upload_page()
    with page_r:
        render_story_page()

    _flush_deferred_error()
    _clear_stale_state_if_needed(uploaded)
    _run_story_pipeline(uploaded, state, make_story)
    _render_footer()


main()
