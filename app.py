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


# =============================================================================
# 2. CACHED MODEL LOADERS
# =============================================================================

@st.cache_resource(show_spinner=False)
def _get_caption_pipeline():
    """Lazy-load BLIP captioner (cached across sessions)."""
    model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL)
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    return {"model": model, "processor": processor}


@st.cache_resource(show_spinner=False)
def _get_story_pipeline():
    """Load TinyStories (preferred) with distilgpt2 fallback."""
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

def img2text(url_or_image):
    """Image → caption using BLIP (bypassing the pipeline wrapper)."""
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
        with urllib.request.urlopen(str(url_or_image)) as resp:
            image = Image.open(BytesIO(resp.read())).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=40)
    text = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    if text:
        text = text[0].upper() + text[1:]
    return text


_STORY_PROMPTS = {
    KID_STORY_MODEL:      "Once upon a time there was a {subject}. ",
    FALLBACK_STORY_MODEL: (
        "Tell a happy bedtime story for a 6 year old child about {subject}. "
        "Use simple words, short sentences, and end with something nice. "
        "Story:\n"
    ),
}


def text2story(text):
    """Caption → story (50–120 words, bedtime style, kid-friendly).

    Notes
    -----
    * We deliberately DO NOT pass ``pad_token_id`` — for TinyStories-33M the
      tokenizer has no dedicated pad token, and passing ``eos_token_id`` as
      pad makes the model stop the moment it emits EOS (mid-sentence).
    * ``max_new_tokens`` is raised to 160 so the model has room to reach a
      natural sentence end.
    * Post-processing repairs ragged endings (missing punctuation, dangling
      articles, single trailing letters like "A.").
    """
    story_gen, model_id = _get_story_pipeline()
    template = _STORY_PROMPTS.get(
        model_id, _STORY_PROMPTS[FALLBACK_STORY_MODEL])

    subject = _extract_subject(text)
    prompt = template.format(subject=subject)

    # --- 1. Generate ---
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": 160,
        "do_sample": True,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
    # Only pass pad_token_id when the tokenizer actually has a pad token.
    tok = getattr(story_gen, "tokenizer", None)
    if tok is not None and getattr(tok, "pad_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tok.pad_token_id

    outputs: list[Any] = story_gen(prompt, **gen_kwargs)
    generated = str(outputs[0].get("generated_text", ""))

    # --- 2. Strip the prompt echo ---
    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    # --- 3. Clean whitespace ---
    generated = re.sub(r"\s+", " ", generated).strip()

    # --- 4. Repair truncation ---
    generated = _repair_story_end(generated)

    # --- 5. Trim to target length on sentence boundary ---
    #     Hard cap of 100 words; never cut mid-sentence.
    generated = _truncate_to_word_count(generated, low=50, high=100)

    return generated


def text2audio(story_text):
    """Story text → MP3 audio bytes."""
    if not story_text:
        return b""
    tts = gTTS(text=story_text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# =============================================================================
# 3a. SMALL TEXT HELPERS
# =============================================================================

_LEADING_ARTICLES = {"a", "an", "the"}
_HAPPY_ENDING = " And they all lived happily ever after. 🌈"


def _extract_subject(caption: str) -> str:
    """Strip only a leading article, keep the caption's natural grammar."""
    words = re.findall(r"[A-Za-z']+", caption)
    if words and words[0].lower() in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words[:8]) or caption


def _truncate_to_word_count(text: str, *, low: int, high: int) -> str:
    """Trim to <= high words, always ending on a COMPLETE sentence.

    Completeness beats length: if trimming to `high` would leave a
    dangling clause, we back off to the last full sentence instead.
    Only pads to `low` words if the model produced less than that.
    """
    if not text:
        return text

    # Split into sentences, keeping the terminal punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return text

    kept: list[str] = []
    count = 0
    for s in sentences:
        n = len(s.split())
        # Would this sentence push us past the hard cap?
        if count + n > high and kept:
            # Stop BEFORE this sentence — never mid-sentence.
            break
        # If it's the very first sentence and it's already over the cap,
        # take it anyway (better one long sentence than nothing).
        kept.append(s)
        count += n

    out = " ".join(kept).strip()

    # Only pad if the model produced too little to begin with.
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
    missing punctuation. Trims back to the last COMPLETE sentence
    rather than stapling a period onto a fragment."""
    if not text:
        return text

    out = text.strip()

    # Drop a lone trailing letter like " A." (mid-word generation cutoff)
    out = _DANGLING_LETTER.sub("", out).rstrip()

    # If the story doesn't end on sentence punctuation, trim back to
    # the last complete sentence — never staple a period onto a fragment.
    if out and not out.endswith((".", "!", "?")):
        last_punct = max(
            out.rfind("."),
            out.rfind("!"),
            out.rfind("?"),
        )
        if last_punct > 0:
            out = out[: last_punct + 1].rstrip()
        # If there's truly no sentence end yet, leave as-is so the
        # downstream truncation can still pick a boundary or pad.

    # Drop a dangling conjunction / article at the very end
    out = _TRAILING_ARTICLE.sub("", out).rstrip()

    # Guarantee a warm, complete-feeling ending for a kids' story
    if not re.search(
        r"(happily ever after|the end)\s*[.!]?\s*$", out, re.IGNORECASE
    ):
        out = out.rstrip() + _HAPPY_ENDING

    return out

# =============================================================================
# 4. UI — Storybook spread (left page = upload, right page = story)
# =============================================================================


# ---------- Floating background emoji + rainbow title ----------
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
<div class="title">
  <div class="title-main">✨ StorySpark ✨</div>
  <div class="title-sub">Show me a picture and I'll tell you a story!</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Session state ----------
for _key, _default in (
    ("story", ""),
    ("audio_bytes", b""),
    ("caption", ""),
    ("celebrated", False),
    ("phase", "idle"),          # idle | working | done
    ("progress_step", 0),       # 0..3 → which loader stage we're on
):
    st.session_state.setdefault(_key, _default)

# ---------- Two facing pages ----------
page_l, page_r = st.columns([1, 1], gap="large")

# =============================================================================
#  LEFT PAGE — Upload
# =============================================================================
with page_l:
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
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        # Disabled while the pipeline is running so the child can't spam it.
        make_story = st.button(
            "Make My Story!",
            type="primary",
            use_container_width=True,
            help="Make some magic ✨",
            key="make_story_btn",
            disabled=(st.session_state.phase == "working"),
        )
    else:
        image = None
        make_story = False
        st.markdown(
            '<div class="wait-hint">'
            '<span class="wait-emoji">☝️</span>'
            '<span>Upload a picture above to see the magic button!</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # --- If the user cleared the upload, wipe any previous story/audio
        #     so the right page doesn't keep showing the old result.
        if st.session_state.story or st.session_state.audio_bytes:
            for k, v in (
                ("story", ""),
                ("audio_bytes", b""),
                ("caption", ""),
                ("celebrated", False),
                ("phase", "idle"),
                ("progress_step", 0),
            ):
                st.session_state[k] = v
            st.rerun()

# =============================================================================
#  RIGHT PAGE — Story OR inline loader OR empty state
# =============================================================================
with page_r:
    story = st.session_state.story
    audio_bytes = st.session_state.audio_bytes
    phase = st.session_state.phase
    step = st.session_state.progress_step

    # -------- State A : pipeline running → inline turning-book loader ----
    if phase == "working":
        _LOADER_STEPS = [
            ("🔎", 0.15, "Looking at your picture"),
            ("📖", 0.45, "Writing your story"),
            ("🎤", 0.80, "Recording the voice"),
            ("✅", 1.00, "All done!"),
        ]
        emoji, pct, note = _LOADER_STEPS[min(step, 3)]

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

    # -------- State B : story ready → show it -------------------------
    elif story:
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

        # ---- Word count chip ----
        _wc = len(story.split())
        st.markdown(
            f'<div class="word-count">'
            f'<span class="wc-emoji">📝</span>'
            f'<span class="wc-num">{_wc}</span>'
            f'<span class="wc-sep">words</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.audio(audio_bytes, format="audio/mp3")

        # --- Download buttons ---------------------------------------------
        # Use raw <a download> links (NOT st.download_button) so that
        # clicking them does NOT trigger a Streamlit script rerun.
        # Streamlit's download_button is a widget — any click causes a
        # rerun, which re-mounts st.audio and resets playback mid-story.
        # Native HTML <a download> just hands the file to the browser.
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        else:
            audio_b64 = ""
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

    # -------- State C : nothing yet → friendly empty state -----------
    else:
        st.markdown(
            '<div class="page-empty">'
            '<div class="empty-emoji">📚</div>'
            '<div class="empty-text">Your story will appear here…</div>'
            '<div class="empty-hint">'
            '🖼️ Upload a picture on the left → '
            '🎨 press the magic button!'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# =============================================================================
# 6. MAIN PIPELINE — two-phase state machine
#    Phase 1: user clicks → phase="working", rerun (loader appears)
#    Phase 2: loader is on screen → run pipeline → phase="done", rerun
# =============================================================================
if make_story and uploaded is not None and st.session_state.phase == "idle":
    # Kick off: flip to "working" so the next paint shows the loader.
    st.session_state.phase = "working"
    st.session_state.progress_step = 0
    st.rerun()

if st.session_state.phase == "working" and uploaded is not None:
    # Re-open the uploaded image for this run (UploadedFile is streamable).
    working_image = Image.open(uploaded)

    try:
        st.session_state.progress_step = 0   # 🔎 Looking…
        caption = img2text(working_image)

        st.session_state.progress_step = 1   # 📖 Writing…
        story = text2story(caption)

        st.session_state.progress_step = 2   # 🎤 Recording…
        audio_bytes = text2audio(story)

        st.session_state.progress_step = 3   # ✅ Done

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
st.markdown(
    "<div class='footer-bar'>Made with 💖 for tiny story-lovers</div>",
    unsafe_allow_html=True,
)
