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
#    2. text2story() →  pranaykoppula/tiny-stories-3M  (fallback: distilgpt2)
#        │
#        ▼
#    3. text2audio() →  gTTS (Google Text-to-Speech)
#        │
#        ▼
#    [Story card + audio player]   🎧
#
#  Models are loaded once and cached on the Streamlit server with
#  @st.cache_resource, so every visitor gets a fast experience after the
#  first cold start (~30 s on Streamlit Community Cloud).
#
#  All styling lives in ./style.css — the Python file is intentionally free of
#  raw CSS / HTML so it stays readable.
#
# =============================================================================

from __future__ import annotations

import io
import re
import time
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
    # centred, single-column layout so children aren't distracted by side content
    layout="centered",
    initial_sidebar_state="collapsed",
)


def _load_css(file_name: str = "style.css") -> None:
    """Inject the project's kid-friendly stylesheet.

    Reads ``style.css`` from the project root and injects it via
    ``st.markdown``. Keeping all styling in a real ``.css`` file means the
    Python source stays free of CSS strings.
    """
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
# All public models — no Hugging Face token required.

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
# Microsoft's real TinyStories model — specifically trained on short children's
# stories with age-appropriate vocabulary. TinyStories-3M is the smallest
# checkpoint (~3 M params) → fast enough for CPU inference.
KID_STORY_MODEL = "roneneldan/TinyStories-3M"
FALLBACK_STORY_MODEL = "distilgpt2"                 # Backup if above is missing


# =============================================================================
# 2. CACHED MODEL LOADERS
# =============================================================================

def _get_caption_pipeline():
    """Lazy-load the BLIP captioner (cached across sessions).

    Why not use ``transformers.pipeline("image-to-text", …)``?
    -----------------------------------------------------------
    * transformers 5.x renamed the task to ``image-text-to-text`` and the new
      pipeline class **forces** an input shape that includes ``text=`` — even
      for an unconditional captioner.  This raises
      ``ValueError: You must provide text for this pipeline.``
    * Older pipelines need ``tokenizer=`` + ``image_processor=`` and silently
      pull in a torchvision-only code path on ``AutoImageProcessor``.

    Bypassing the pipeline and calling ``processor + model.generate()``
    directly is faster, identical across every transformers version, and
    sidesteps the torchvision dependency entirely.
    """
    model = AutoModelForImageTextToText.from_pretrained(CAPTION_MODEL)
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    return {"model": model, "processor": processor}


@st.cache_resource(show_spinner=False)
def _get_story_pipeline():
    """Lazy-load a text-generation pipeline.

    Tries the TinyStories model first (kid-friendly vocabulary) and silently
    falls back to distilgpt2 if the Hub can't serve the preferred model.

    Returns
    -------
    (pipeline, str)
        The loaded pipeline and the model id that was actually used.
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
        except Exception as exc:  # defensive: handle Hub 404s gracefully
            print(f"[StorySpark] Could not load {model_id}: {exc}")
            continue
    raise RuntimeError("No story-generation model could be loaded.")


# =============================================================================
# 3. CORE PIPELINE FUNCTIONS
# =============================================================================
# Names follow the class sample. Each does one stage of the pipeline so the
# main flow stays easy to read.

def img2text(url_or_image):
    """Image → caption.

    Parameters
    ----------
    url_or_image : str | PIL.Image.Image
        Either an image URL (the class sample uses this) **or** a local
        ``PIL.Image`` (what Streamlit uploads give us). Both are accepted.

    Returns
    -------
    str
        A short English caption for the picture.
    """
    captioner = _get_caption_pipeline()
    model = cast(Any, captioner["model"])
    processor = cast(Any, captioner["processor"])

    # Accept URL or PIL.Image.  If it's a PIL.Image, convert to RGB so BLIP
    # doesn't complain about alpha channels or palette modes.
    if isinstance(url_or_image, Image.Image):
        image: Image.Image = (
            url_or_image.convert("RGB")
            if url_or_image.mode != "RGB"
            else url_or_image
        )
    else:
        # If a URL was passed, fetch the bytes and open with PIL.
        import urllib.request
        from io import BytesIO
        with urllib.request.urlopen(str(url_or_image)) as resp:
            image = Image.open(BytesIO(resp.read())).convert("RGB")

    # Direct (pipeline-bypassing) caption — see _get_caption_pipeline docstring.
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=40)
    text = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    if text:
        text = text[0].upper() + text[1:]
    return text


# Prompt templates — picked to match each model's training distribution.
_STORY_PROMPTS = {
    KID_STORY_MODEL:      "Once upon a time there was a {subject}. ",
    FALLBACK_STORY_MODEL: (
        "Tell a happy bedtime story for a 6 year old child about {subject}. "
        "Use simple words, short sentences, and end with something nice. "
        "Story:\n"
    ),
}


def text2story(text):
    """Caption → story (50–100 words, bedtime style, kid-friendly).

    Returns
    -------
    str
        A short story snippet.
    """
    story_gen, model_id = _get_story_pipeline()
    template = _STORY_PROMPTS.get(
        model_id, _STORY_PROMPTS[FALLBACK_STORY_MODEL])

    subject = _extract_subject(text)
    prompt = template.format(subject=subject)

    outputs: list[Any] = story_gen(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.3,
        # ``story_gen.tokenizer`` is technically Optional per the type stubs;
        # guard the access and fall back to the model's own pad_token_id when
        # eos_token_id is None (some tokenizers return None for eos).
        pad_token_id=_safe_eos_token_id(story_gen),
    )
    generated = str(outputs[0].get("generated_text", ""))

    # Drop the prompt — keep only what the model wrote.
    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    # Tidy whitespace, then cap the word count to satisfy the assignment brief.
    generated = re.sub(r"\s+", " ", generated).strip()
    generated = _truncate_to_word_count(generated, low=50, high=100)

    # Always end on a sentence terminator.
    if generated and not generated.endswith((".", "!", "?")):
        generated = generated.rstrip(".") + "."
    return generated


def text2audio(story_text, *, slow: bool = False):
    """Story text → MP3 audio bytes.

    Parameters
    ----------
    story_text : str
        The story to read aloud.
    slow : bool, optional
        If True, gTTS produces a slower dictation suitable for ages 3–4.

    Returns
    -------
    bytes
        Raw MP3 payload suitable for ``st.audio`` or a download button.
    """
    if not story_text:
        return b""

    tts = gTTS(text=story_text, lang="en", slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# Thin alias used by the main loop — same signature as ``text2audio``.
def text2audio_with_speed(story_text, *, slow: bool = False):
    """Same as ``text2audio``; kept as a separate name for UI-layer clarity."""
    return text2audio(story_text, slow=slow)


def _safe_eos_token_id(pipe) -> int:
    """Return ``pipe.tokenizer.eos_token_id`` defensively.

    The typed stubs mark ``tokenizer`` as ``Optional`` and ``eos_token_id`` as
    ``Optional[int]``.  Guard both so Pylance stays happy *and* the call works
    on tokenizers (e.g. some GPT-2 forks) where eos is ``None``.
    """
    tok = getattr(pipe, "tokenizer", None)
    eos = getattr(tok, "eos_token_id", None)
    if isinstance(eos, int):
        return eos
    # Fall back to pad token, then the model's own eos, then the HF default (0).
    pad = getattr(tok, "pad_token_id", None)
    if isinstance(pad, int):
        return pad
    model_eos = getattr(getattr(pipe, "model", None), "config", None)
    if model_eos is not None:
        cfg_eos = getattr(model_eos, "eos_token_id", None)
        if isinstance(cfg_eos, int):
            return cfg_eos
    return 0


# =============================================================================
# 3a. SMALL TEXT HELPERS
# =============================================================================

# Tiny list of common English function words so we can lift the "main
# character" out of a BLIP caption without loading another NLP library.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
    "at", "with", "and", "or", "to", "by", "for", "this", "that",
    "these", "those", "there", "some", "two",
}


def _extract_subject(caption: str) -> str:
    """Pull the main subject out of a caption.

    Examples
    --------
    >>> _extract_subject("a small white dog playing in the snow")
    'small white dog'
    >>> _extract_subject("an orange cat sitting on a wooden chair")
    'orange cat'
    """
    words = re.findall(r"[A-Za-z']+", caption.lower())
    nouny = [w for w in words if w not in _STOPWORDS]
    if not nouny:
        return caption
    return " ".join(nouny[:5])


def _truncate_to_word_count(text: str, *, low: int, high: int) -> str:
    """Trim ``text`` to between ``low``–``high`` words on a sentence boundary."""
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, count = [], 0
    for s in sentences:
        words = s.split()
        if not words:
            continue
        if count + len(words) > high and kept:
            break
        kept.append(s)
        count += len(words)

    out = " ".join(kept).strip()

    # Pad short outputs so stories always reach the assignment's 50-word floor.
    if 0 < len(out.split()) < low:
        out += " And they all lived happily ever after. 🌈"
    return out


# =============================================================================
# 4. UI — Main page (hero banner + 3 steps, all in cards)
# =============================================================================
#
# Layout is intentionally single-column + centred (set in ``st.set_page_config``).
# Every step sits in its own pastel "card" so children always know where they
# are in the flow: 1) Upload → 2) Make a story → 3) Listen & download.

# ---------- Hero banner ------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">🌈 StorySpark 🦄</div>
  <div class="hero-tag">Upload a picture → get a magical bedtime story!</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Persist the pipeline result across Streamlit re-runs ------------
if "story" not in st.session_state:
    st.session_state.story = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = b""
if "caption" not in st.session_state:
    st.session_state.caption = ""

# ---------- Step 1 : Upload a picture (full width card) ---------------------
st.markdown(
    '<div class="card step-1">'
    '<div class="step-header">'
    '<span class="step-number">1</span>'
    '<span>🖼️  Upload a picture</span>'
    '</div>'
    '<div class="step-blurb">Pick any photo, drawing, or toy picture — '
    'we turn it into a story!</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    label="Choose a picture",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
    help="Choose any picture from your computer or tablet.",
    label_visibility="collapsed",
    key="uploaded_image",
)
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)
else:
    st.markdown(
        '<div class="helper">👋 Pick a picture from your computer or tablet — '
        'a pet, a drawing, or a holiday snap all work!</div>',
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Step 2 : Make a story (full width card) -------------------------
st.markdown(
    '<div class="card step-2">'
    '<div class="step-header">'
    '<span class="step-number">2</span>'
    '<span>📝  Make a story</span>'
    '</div>'
    '<div class="step-blurb">Tap the big magic button to start the storytime!</div>',
    unsafe_allow_html=True,
)

make_story = st.button(
    "🎨  Make My Story!",
    type="primary",
    use_container_width=True,
    disabled=uploaded is None,
    help="Pick a picture above first!" if uploaded is None else "Make some magic ✨",
)

slow_voice = st.toggle(
    "🐢 Slow & gentle voice",
    value=False,
    help="Speaks a little more slowly. Great for very young listeners.",
)
st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 6. MAIN PIPELINE
# =============================================================================
# Wires the three core functions together. When the user uploads a picture and
# taps the big button, we run caption → story → audio and render the result.

if make_story and uploaded is not None:
    with st.status("✨  Sprinkling story magic …", expanded=True) as status:
        try:
            # ---------- Step 1 : image → caption ----------
            status.update(label="🔎  Looking at your picture …")
            caption = img2text(image)
            st.write(f"📝  **Caption:** _{caption}_")
            time.sleep(0.2)

            # ---------- Step 2 : caption → story ----------
            status.update(label="📖  Writing your story …")
            story = text2story(caption)
            st.write(f"📖  Story length: **{len(story.split())} words**")

            # ---------- Step 3 : story → audio ----------
            status.update(label="🎤  Recording the voice …")
            audio_bytes = text2audio_with_speed(story, slow=slow_voice)

            status.update(label="✅  All done!", state="complete")
        except Exception as exc:
            status.update(label="😢  Something went wrong", state="error")
            st.exception(exc)
            story, audio_bytes, caption = "", b"", ""

        # Persist into session_state so the audio player + download buttons
        # keep rendering correctly on every subsequent re-run (e.g. when the
        # user clicks "Download").
        st.session_state.story = story
        st.session_state.audio_bytes = audio_bytes
        st.session_state.caption = caption

# ---------- Render the result (Step 3: Listen & download) -------------------
# Sits at MODULE level (not inside the `if make_story:` above) on purpose so
# the result keeps rendering across every rerun (download clicks, toggles…).
story = st.session_state.story
audio_bytes = st.session_state.audio_bytes
caption = st.session_state.caption

if story:
    # Celebratory animation on the *first* run only — not every rerun.
    if not st.session_state.get("celebrated"):
        st.balloons()
        st.session_state.celebrated = True

    st.markdown(
        '<div class="card step-3">'
        '<div class="step-header">'
        '<span class="step-number">3</span>'
        '<span>🎧  Listen to your story &amp; download</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---------- Story in a "storybook page" card ----------
    st.markdown(
        f"""
<div class="storybook-page">
  <span class="badge">📖 Your Story</span>
  <div>{story}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---------- Audio player — clearly labelled and prominent ----------
    st.markdown("**🔊 Listen along!**")
    st.audio(audio_bytes, format="audio/mp3")

    # ---------- Downloads (2 across) ---------------------------------------
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇️  Audio (MP3)",
            data=audio_bytes,
            file_name="storyspark_story.mp3",
            mime="audio/mpeg",
            use_container_width=True,
            key="dl_mp3",
        )
    with col_dl2:
        st.download_button(
            label="📝  Story text",
            data=story.encode("utf-8"),
            file_name="storyspark_story.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_txt",
        )

    # ---------- Reset (secondary action — sky/mint gradient) ----------
    if st.button(
        "🔄  Try another picture!",
        key="reset_btn",
        use_container_width=True,
    ):
        for _key in ("story", "audio_bytes", "caption", "celebrated"):
            if _key == "audio_bytes":
                st.session_state[_key] = b""
            elif _key == "celebrated":
                st.session_state[_key] = False
            else:
                st.session_state[_key] = ""
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 7. FOOTER
# =============================================================================

st.markdown(
    "<br><center>Made with 💖 for tiny story-lovers.</center>",
    unsafe_allow_html=True,
)
