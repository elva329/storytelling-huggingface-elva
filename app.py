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

import io
import re
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, cast

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
    """Caption → story (50–100 words, bedtime style, kid-friendly)."""
    story_gen, model_id = _get_story_pipeline()
    template = _STORY_PROMPTS.get(
        model_id, _STORY_PROMPTS[FALLBACK_STORY_MODEL])

    subject = _extract_subject(text)
    prompt = template.format(subject=subject)

    outputs: list[Any] = story_gen(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.15,
        pad_token_id=story_gen.tokenizer.eos_token_id,
    )
    generated = str(outputs[0].get("generated_text", ""))

    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    generated = re.sub(r"\s+", " ", generated).strip()
    generated = _truncate_to_word_count(generated, low=50, high=100)

    if generated and not generated.endswith((".", "!", "?")):
        generated = generated.rstrip(".") + "."
    return generated


def text2audio(story_text, *, slow: bool = False):
    """Story text → MP3 audio bytes."""
    if not story_text:
        return b""
    tts = gTTS(text=story_text, lang="en", slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _resolve_pad_token_id(pipe) -> int:
    """Safe pad_token_id for generation across tokenizer variants."""
    tok = getattr(pipe, "tokenizer", None)
    for attr in ("pad_token_id", "eos_token_id"):
        val = getattr(tok, attr, None)
        if isinstance(val, int):
            return val
    cfg = getattr(getattr(pipe, "model", None), "config", None)
    if cfg is not None:
        cfg_eos = getattr(cfg, "eos_token_id", None)
        if isinstance(cfg_eos, int):
            return cfg_eos
    return 0


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
    """Trim to <= high words on a sentence boundary, pad to >= low words."""
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, count = [], 0
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

    # ---- Only show the image + toggle + button once a picture is picked ----
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        st.markdown(
            '<div class="voice-row">'
            '<span class="voice-emoji">🐢</span>'
            '<span class="voice-text">Slow &amp; gentle voice</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        slow_voice = st.toggle(
            "Slow & gentle voice",
            value=False,
            help="Great for very young listeners.",
            label_visibility="collapsed",
        )
        make_story = st.button(
            "🎨  Make My Story!",
            type="primary",
            use_container_width=True,
            help="Make some magic ✨",
            key="make_story_btn",
        )
    else:
        image = None
        slow_voice = False
        make_story = False
        st.markdown(
            '<div class="wait-hint">'
            '<span class="wait-emoji">☝️</span>'
            '<span>Upload a picture above to see the magic button!</span>'
            '</div>',
            unsafe_allow_html=True,
        )

# =============================================================================
#  RIGHT PAGE — Story
# =============================================================================
with page_r:
    story = st.session_state.story
    audio_bytes = st.session_state.audio_bytes

    if story:
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

        st.audio(audio_bytes, format="audio/mp3")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "🔊  Save the voice",
                data=audio_bytes,
                file_name="storyspark_story.mp3",
                mime="audio/mpeg",
                use_container_width=True,
                key="dl_mp3",
            )
        with dl2:
            st.download_button(
                "📜  Save the story",
                data=story.encode("utf-8"),
                file_name="storyspark_story.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_txt",
            )

        if st.button(
            "🎈  One more story!",
            key="reset_btn",
            use_container_width=True,
        ):
            for k, v in (
                ("story", ""),
                ("audio_bytes", b""),
                ("caption", ""),
                ("celebrated", False),
            ):
                st.session_state[k] = v
            st.rerun()

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
# 6. MAIN PIPELINE — custom on-theme progress overlay
# =============================================================================
if make_story and uploaded is not None:
    progress_slot = st.empty()

    def _show_progress(emoji: str, pct: float, note: str) -> None:
        """Render an on-theme progress card in a placeholder."""
        progress_slot.markdown(
            f"""
<div class="progress-card">
  <div class="progress-emoji">{emoji}</div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{pct*100:.0f}%"></div>
  </div>
  <div class="progress-text">{note}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    try:
        _show_progress("🔎", 0.15, "Looking at your picture…")
        caption = img2text(image)

        _show_progress("📖", 0.45, "Writing your story…")
        story = text2story(caption)

        _show_progress("🎤", 0.80, "Recording the voice…")
        audio_bytes = text2audio(story, slow=slow_voice)

        _show_progress("✅", 1.00, "All done!")
        time.sleep(0.4)
        progress_slot.empty()

    except Exception as exc:
        progress_slot.empty()
        st.error(f"😢 Oops! Something went wrong: {exc}")
        story, audio_bytes, caption = "", b"", ""

    st.session_state.story = story
    st.session_state.audio_bytes = audio_bytes
    st.session_state.caption = caption
    st.session_state.celebrated = False
    st.rerun()

# =============================================================================
# 7. FOOTER — fixed bottom bar
# =============================================================================
st.markdown(
    "<div class='footer-bar'>Made with 💖 for tiny story-lovers</div>",
    unsafe_allow_html=True,
)
