# =============================================================================
#  📖✨ StorySpark — A Storytelling Application for Kids (ages 3–10) ✨📖
# =============================================================================
#
#  Course : ISOM5240 — Deep Learning Business Applications with Python
#  Author : Elva
#  Stack  : Streamlit · Hugging Face Transformers · gTTS
#
#  Pipeline
#  --------
#    [Image upload]
#        │
#        ▼
#    1. img2text()  →  Salesforce/blip-image-captioning-base
#        │
#        ▼
#    2. text2story() →  roneneldan/TinyStories-33M  (fallback: distilgpt2)
#        │
#        ▼
#    3. text2audio() →  gTTS (Google Text-to-Speech)
#        │
#        ▼
#    [Storybook spread with audio player]   🎧
#
#  All styling lives in ./style.css.
#
# =============================================================================

from __future__ import annotations

import base64
import io
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import streamlit as st
from PIL import Image

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
)

from gtts import gTTS


# =============================================================================
# 0. PAGE CONFIG & EXTERNAL STYLE
# =============================================================================

st.set_page_config(
    page_title="StorySpark — Make a Story!",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _load_css(file_name: str = "style.css") -> None:
    """Inject the project's kid-friendly stylesheet."""
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


_load_css()


# =============================================================================
# 1. MODEL CONFIGURATION
# =============================================================================

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
KID_STORY_MODEL = "roneneldan/TinyStories-33M"
FALLBACK_STORY_MODEL = "distilgpt2"

# Story-gen prompts per model. TinyStories just needs a seed sentence;
# distilgpt2 needs an explicit instruction to write a kid's bedtime story.
_STORY_PROMPTS: dict[str, str] = {
    KID_STORY_MODEL: "Once upon a time there was a {subject}. ",
    FALLBACK_STORY_MODEL: (
        "Tell a happy bedtime story for a 6 year old child about {subject}. "
        "Use simple words, short sentences, and end with something nice. "
        "Story:\n"
    ),
}

# Maps the pipeline phase to which quest-path step should be highlighted.
#   idle    → step 1 (pick a picture)
#   working → step 2 (make my story)
#   done    → step 3 (listen & save)
PHASE_TO_ACTIVE_STEP: dict[str, int] = {
    "idle": 1,
    "working": 2,
    "done": 3,
}

# Default values for every key we put in st.session_state. Used both to
# initialise the session on first run and to reset it when the user
# removes their upload.
DEFAULT_SESSION_STATE: dict[str, Any] = {
    "story": "",
    "audio_bytes": b"",
    "caption": "",
    "celebrated": False,
    "phase": "idle",            # one of: idle | working | done
    "progress_step": 0,
}

# Steps shown by the inline turning-book loader while the pipeline runs.
# Each tuple is (emoji, fractional progress, status text).
LOADER_STEPS: list[tuple[str, float, str]] = [
    ("🔎", 0.15, "Looking at your picture"),
    ("📖", 0.45, "Writing your story"),
    ("🎤", 0.80, "Recording the voice"),
    ("✅", 1.00, "All done!"),
]


# =============================================================================
# 2. CACHED MODEL LOADERS
# =============================================================================

@st.cache_resource(show_spinner=False)
def _get_caption_pipeline() -> dict[str, Any]:
    """Lazy-load the BLIP captioner (cached across sessions)."""
    model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL)
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    return {"model": model, "processor": processor}


@st.cache_resource(show_spinner=False)
def _get_story_pipeline() -> tuple[Any, str]:
    """Load TinyStories (preferred) with distilgpt2 as a fallback.

    Returns the (text_generation_pipeline, model_id) tuple of whichever
    model loaded first; raises RuntimeError if none succeed.
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
        except Exception as exc:
            print(f"[StorySpark] Could not load {model_id}: {exc}")
            continue
    raise RuntimeError("No story-generation model could be loaded.")


# =============================================================================
# 3. CORE PIPELINE FUNCTIONS
# =============================================================================

def img2text(url_or_image: str | Image.Image) -> str:
    """Image → short caption using the BLIP image-text-to-text model.

    Parameters
    ----------
    url_or_image : str | PIL.Image.Image
        Either a PIL Image (typical case from the Streamlit uploader)
        or a URL string pointing at a remote image.
    """
    captioner = _get_caption_pipeline()
    model = cast(Any, captioner["model"])
    processor = cast(Any, captioner["processor"])

    if isinstance(url_or_image, Image.Image):
        image: Image.Image = (
            url_or_image.convert("RGB")
            if url_or_image.mode != "RGB"
            else url_or_image
        )
    else:
        with urllib.request.urlopen(str(url_or_image)) as response:
            image = Image.open(BytesIO(response.read())).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=40)
    text = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    if text:
        text = text[0].upper() + text[1:]
    return text


def text2story(caption: str) -> str:
    """Caption → kid-friendly bedtime story (50–100 words).

    Emojis and stray quotes are stripped, the end is repaired so the
    story finishes on a complete sentence, and the result is trimmed
    to the assignment's 50–100 word target.
    """
    story_gen, model_id = _get_story_pipeline()
    template = _STORY_PROMPTS.get(
        model_id, _STORY_PROMPTS[FALLBACK_STORY_MODEL])

    subject = _extract_subject(caption)
    prompt = template.format(subject=subject)

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

    outputs: list[Any] = story_gen(prompt, **gen_kwargs)
    generated = str(outputs[0].get("generated_text", ""))

    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    generated = re.sub(r"\s+", " ", generated).strip()
    generated = _strip_emojis(generated)       # strip early
    generated = _repair_story_end(generated)
    generated = _truncate_to_word_count(generated, low=50, high=100)
    generated = _strip_emojis(generated)       # strip final

    return generated


def text2audio(story_text: str) -> bytes:
    """Story text → MP3 audio bytes via Google Text-to-Speech.

    Emojis and quotes are stripped first so the voice never tries to
    pronounce them. Returns an empty bytes object if the input is empty.
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


# =============================================================================
# 3a. SMALL TEXT HELPERS
# =============================================================================

_LEADING_ARTICLES = {"a", "an", "the"}
_HAPPY_ENDING = " And they all lived happily ever after."

# Matches most emoji codepoints: pictographs, symbols, dingbats,
# variation selectors, ZWJ sequences, and skin-tone modifiers.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"   # Symbols & Pictographs
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F680-\U0001F6FF"   # Transport & Map
    "\U0001F700-\U0001F77F"   # Alchemical
    "\U0001F780-\U0001F7FF"   # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"   # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"   # Supplemental Symbols & Pictographs
    "\U0001FA00-\U0001FA6F"   # Chess
    "\U0001FA70-\U0001FAFF"   # Symbols & Pictographs Extended-A
    "\U00002600-\U000026FF"   # Misc symbols
    "\U00002700-\U000027BF"   # Dingbats
    "\U0001F1E6-\U0001F1FF"   # Regional Indicator (flags)
    "\U0000FE00-\U0000FE0F"   # Variation Selectors
    "\U0001F3FB-\U0001F3FF"   # Skin tone modifiers
    "\U0000200D"              # ZWJ
    "\U00002B00-\U00002BFF"   # Misc Symbols and Arrows
    "]+",
    flags=re.UNICODE,
)

# Double quotes (straight + curly) and backticks. Apostrophes (') are
# NOT in this class — we want to preserve contractions like "it's".
_QUOTE_RE = re.compile(r'["“”„«»`]')


def _strip_emojis(text: str) -> str:
    """Remove emojis, quotes, and other characters that gTTS would
    pronounce awkwardly. Tidy up the resulting spacing."""
    if not text:
        return text

    # 1. Remove emojis
    cleaned = _EMOJI_RE.sub("", text)

    # 2. Remove straight/curly double quotes and backticks
    cleaned = _QUOTE_RE.sub("", cleaned)

    # 3. Collapse double spaces + trim spacing around punctuation
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)

    # 4. Ensure a space after a colon/comma when it was glued to a word
    cleaned = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", cleaned)

    return cleaned.strip()


def _extract_subject(caption: str) -> str:
    """Strip only a leading article, keep the caption's natural grammar."""
    words = re.findall(r"[A-Za-z']+", caption)
    if words and words[0].lower() in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words[:8]) or caption


def _truncate_to_word_count(text: str, *, low: int, high: int) -> str:
    """Trim to <= high words, always ending on a COMPLETE sentence."""
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
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

    safety = 0
    while 0 < len(out.split()) < low and safety < 3:
        out += _HAPPY_ENDING
        safety += 1

    return out


_TRAILING_ARTICLE = re.compile(
    r"\b(a|an|the|and|but|or|so|because|with|for|to|of|in|on|at|"
    r"was|were|is|are|be|been|being)\s*$",
    re.IGNORECASE,
)
_DANGLING_LETTER = re.compile(r"\s+[A-Z]\.\s*$")


def _repair_story_end(text: str) -> str:
    """Fix ragged endings: dangling articles, trailing single letters,
    missing punctuation. Trims back to the last COMPLETE sentence."""
    if not text:
        return text

    out = text.strip()
    out = _DANGLING_LETTER.sub("", out).rstrip()

    if out and not out.endswith((".", "!", "?")):
        last_punct = max(
            out.rfind("."),
            out.rfind("!"),
            out.rfind("?"),
        )
        if last_punct > 0:
            out = out[: last_punct + 1].rstrip()

    out = _TRAILING_ARTICLE.sub("", out).rstrip()

    if not re.search(
        r"(happily ever after|the end)\s*[.!]?\s*$", out, re.IGNORECASE
    ):
        out = out.rstrip() + _HAPPY_ENDING

    return out


# =============================================================================
# 3b. SESSION-STATE HELPERS
# =============================================================================

def _init_session_state() -> None:
    """Populate st.session_state with the default values on first run."""
    for key, default in DEFAULT_SESSION_STATE.items():
        st.session_state.setdefault(key, default)


def _reset_session_state() -> None:
    """Reset every StorySpark session key back to its default value."""
    for key, default in DEFAULT_SESSION_STATE.items():
        st.session_state[key] = default


# =============================================================================
# 4. UI — small HTML-fragment helpers
# =============================================================================

# Quest-path step definitions used by _render_quest_path(). Tuples are
# (icon, title, sub). The '&' in "Listen & save" is pre-escaped for HTML.
_QUEST_STEPS: list[tuple[str, str, str]] = [
    ("🎨", "Pick a picture",   "PNG · JPG · WEBP · up to 25 MB"),
    ("✨", "Make my story",    "Tap the big pink button"),
    ("🎧", "Listen &amp; save", "Hear your story come alive"),
]


def _quest_step_classes(step_num: int, active_step: int) -> str:
    """Return the CSS class string for a quest-path step at index n."""
    if step_num < active_step:
        return "quest-step done"
    if step_num == active_step:
        return "quest-step active"
    return "quest-step"


def _quest_badge_text(step_num: int, active_step: int) -> str:
    """Return the badge text for a quest-path step (✓ if done, else its number)."""
    return "✓" if step_num < active_step else str(step_num)


def _render_floating_scene() -> None:
    """Render the fixed-position floating background emoji scene."""
    st.markdown(
        """
<div class="scene">
  <span class="float f1">🌈</span>
  <span class="float f2">⭐</span>
  <span class="float f3">🎈</span>
  <span class="float f4">☁️</span>
  <span class="float f5">✨</span>
  <span class="float f6">🦄</span>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_title() -> None:
    """Render the rainbow 'StorySpark' title and tagline."""
    st.markdown(
        """
<div class="title">
  <div class="title-main">✨ StorySpark ✨</div>
  <div class="title-sub">Show me a picture and I'll tell you a story!</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_quest_path(active_step: int) -> None:
    """Render the 3-step 'Pick → Make → Listen' quest path shown in the empty state."""
    blocks: list[str] = []
    for n, (icon, title, sub) in enumerate(_QUEST_STEPS, start=1):
        blocks.append(
            f'<div class="{_quest_step_classes(n, active_step)}">'
            f'<div class="quest-badge">{_quest_badge_text(n, active_step)}</div>'
            f'<div class="quest-icon">{icon}</div>'
            f'<div class="quest-text">'
            f'<div class="quest-title">{title}</div>'
            f'<div class="quest-sub">{sub}</div>'
            f'</div>'
            f'</div>'
        )
        # A connector goes after every step EXCEPT the last.
        if n < len(_QUEST_STEPS):
            blocks.append('<div class="quest-connector"></div>')

    st.markdown(
        f'<div class="quest-path">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def _render_progress_loader(step: int) -> None:
    """Render the inline turning-book loader with progress bar and emoji."""
    emoji, pct, note = LOADER_STEPS[min(step, len(LOADER_STEPS) - 1)]
    st.markdown(
        f"""
<div class="progress-inline">
  <div class="turning-book" aria-hidden="true">
    <div class="base">
      <div class="left"></div>
      <div class="right"></div>
    </div>
    <div class="spine"></div>
    <div class="page">
      <span class="line l1"></span><span class="line l2"></span>
      <span class="line l3"></span><span class="line l4"></span>
    </div>
  </div>
  <div class="progress-emoji">{emoji}</div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{pct*100:.0f}%"></div>
  </div>
  <div class="progress-text">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_download_buttons(audio_bytes: bytes, story: str) -> None:
    """Render two raw <a download> links for the audio (MP3) and story (TXT).

    We use raw links rather than st.download_button so that clicking
    them does NOT trigger a Streamlit rerun and interrupt audio playback.
    """
    audio_b64 = base64.b64encode(audio_bytes).decode(
        "ascii") if audio_bytes else ""
    story_url = quote(story, safe="")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown(
            f'<div class="ss-dl-cell">'
            f'<a class="ss-dl-link ss-dl-voice" '
            f'href="data:audio/mp3;base64,{audio_b64}" '
            f'download="storyspark_story.mp3">'
            f'Save the voice</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="ss-dl-cell">'
            f'<a class="ss-dl-link ss-dl-story" '
            f'href="data:text/plain;charset=utf-8,{story_url}" '
            f'download="storyspark_story.txt">'
            f'Save the story</a>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_story_output(story: str, audio_bytes: bytes) -> None:
    """Render the completed story: heading, text, word count, audio, downloads."""
    if not st.session_state.celebrated:
        st.balloons()
        st.session_state.celebrated = True

    st.markdown(
        '<div class="page-heading">'
        '<span class="page-emoji">📖</span>'
        '<span>Here comes your story!</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="story-text">{story}</div>',
        unsafe_allow_html=True,
    )

    word_count = len(story.split())
    st.markdown(
        f'<div class="word-count">'
        f'<span class="wc-emoji">📝</span>'
        f'<span class="wc-num">{word_count}</span>'
        f'<span class="wc-sep">words</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.audio(audio_bytes, format="audio/mp3")
    _render_download_buttons(audio_bytes, story)


def _render_empty_state() -> None:
    """Render Ollie the Story Owl plus the original guidance text."""
    st.markdown(
        """
<div class="page-empty">

  <!-- Ollie the Story Owl -->
  <div class="story-guide" aria-hidden="true">
    <div class="guide-body">
      <div class="guide-eyes">
        <span class="eye"><span class="pupil"></span></span>
        <span class="eye"><span class="pupil"></span></span>
      </div>
      <div class="guide-beak"></div>
    </div>
    <div class="guide-wing left"></div>
    <div class="guide-wing right"></div>
  </div>

  <!-- Speech bubble -->
  <div class="guide-bubble">
    <span class="guide-greet">Hi, story-maker!</span>
    <span class="guide-ask">
      I'm Ollie the Story Owl 🦉<br>
      Show me a picture and I'll<br>
      tell you a tale just for you…
    </span>
  </div>

  <!-- Original guidance text, restored below Ollie -->
  <div class="empty-original">
    <div class="empty-text">Your story will appear here…</div>
    <div class="empty-hint">
      🖼️ Upload a picture on the left → 🎨 press the magic button!
    </div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# 5. UI — page-level renderers
# =============================================================================

def render_upload_page() -> tuple[Any, bool]:
    """Render the LEFT (upload) page.

    Returns
    -------
    uploaded : Any
        The currently uploaded file (Streamlit UploadedFile), or None if
        nothing has been uploaded yet.
    make_story : bool
        True only on the run where the user just clicked the
        'Make My Story!' button.
    """
    st.markdown(
        '<div class="page-heading">'
        '<span class="page-emoji">🎨</span>'
        '<span>Put your picture here!</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        label="Upload a picture",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="uploaded_image",
    )

    if uploaded is not None:
        # Post-upload state: show the picture + the magic 'Make My Story!' button.
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        if st.session_state.phase == "working":
            # Visually grey out the uploader while the pipeline is running.
            st.markdown(
                """
<style>
[data-testid="stFileUploader"] {
    pointer-events: none;
    opacity: 0.55;
}
</style>
""",
                unsafe_allow_html=True,
            )

        make_story = st.button(
            "Make My Story!",
            type="primary",
            use_container_width=True,
            help="Tap to make a story from your picture ✨",
            key="make_story_btn",
            disabled=(st.session_state.phase == "working"),
        )
        return uploaded, make_story

    # Empty state: show the quest path and reset any stale story/audio.
    active_step = PHASE_TO_ACTIVE_STEP.get(st.session_state.phase, 1)
    _render_quest_path(active_step)

    if st.session_state.story or st.session_state.audio_bytes:
        # The user removed their upload — drop any leftover story/audio so
        # we don't show stale content on the next run.
        _reset_session_state()
        st.rerun()

    return uploaded, False


def render_story_page() -> None:
    """Render the RIGHT (story) page: turning-book loader, finished story,
    or the Ollie empty state — depending on the current phase."""
    story = st.session_state.story
    audio_bytes = st.session_state.audio_bytes
    phase = st.session_state.phase
    step = st.session_state.progress_step

    if phase == "working":
        _render_progress_loader(step)
    elif story:
        _render_story_output(story, audio_bytes)
    else:
        _render_empty_state()


# =============================================================================
# 6. MAIN PIPELINE — two-phase state machine
# =============================================================================

def _run_story_pipeline(uploaded: Any, make_story: bool) -> None:
    """Drive the image → story → audio state machine.

    Runs in two phases via Streamlit reruns:

    * Phase 1 — user clicks 'Make My Story!': flip phase to 'working'
      and rerun, so the loader UI is shown immediately.
    * Phase 2 — phase is 'working': execute the pipeline step by step,
      then either complete (flip to 'done') or fall back to 'idle' on
      failure.
    """
    # ---- Phase 1: user just clicked the magic button ----
    if make_story and uploaded is not None and st.session_state.phase == "idle":
        st.session_state.phase = "working"
        st.session_state.progress_step = 0
        st.rerun()

    # ---- Phase 2: actually run the pipeline ----
    if st.session_state.phase == "working" and uploaded is not None:
        working_image = Image.open(uploaded)

        try:
            st.session_state.progress_step = 0
            caption = img2text(working_image)

            st.session_state.progress_step = 1
            story = text2story(caption)

            st.session_state.progress_step = 2
            audio_bytes = text2audio(story)

            st.session_state.progress_step = 3

        except Exception as exc:
            st.session_state.phase = "idle"
            st.session_state.progress_step = 0
            st.error(f"😢 Oops! Something went wrong: {exc}")
            story, audio_bytes, caption = "", b"", ""

        st.session_state.story = story
        st.session_state.audio_bytes = audio_bytes
        st.session_state.caption = caption
        st.session_state.celebrated = False
        st.session_state.phase = "done" if story else "idle"
        st.session_state.progress_step = 0
        st.rerun()


# =============================================================================
# 7. FOOTER — fixed bottom bar
# =============================================================================

def _render_footer() -> None:
    """Render the fixed-position footer bar at the bottom of the page."""
    st.markdown(
        "<div class='footer-bar'>Made with 💖 for tiny story-lovers</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. ENTRY POINT
# =============================================================================

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

    _run_story_pipeline(uploaded, make_story)
    _render_footer()


main()
