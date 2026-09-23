# 📖 StorySpark — A Storytelling App for Kids (ages 3–10)

> **Course:** ISOM5240 — Deep Learning Business Applications with Python
> **Assignment:** Individual Assignment
> **Author:** Elva

**Live app:** `https://storytelling-huggingface-elva.streamlit.app/`

StorySpark turns any picture into a short, 50–100-word bedtime story that the app then reads aloud. Built with Hugging Face Transformers and Streamlit, designed for ages 3–10.

---

## 🎯 Objective

Build a Python storytelling app that (1) accepts an uploaded image, (2) generates a 50–100-word narrative from it, (3) converts the story to audio, and (4) runs as an interactive Streamlit app deployable on Streamlit Cloud.

---

## ✨ Features

- Upload a picture (PNG / JPG / WEBP, ≤ 5 MB)
- Automatic image captioning (BLIP)
- 50–100-word story generation (SmolLM2-360M-Instruct, TinyStories fallback)
- Text-to-speech playback (gTTS)
- Download links for both story (`.txt`) and voice (`.mp3`)
- Kid-friendly UI: rainbow title, big buttons, animated loader, Ollie the Story Owl guide

---

## 🛠 Technologies

| Layer | Tool |
|---|---|
| UI | Streamlit + custom CSS |
| Captioning | Hugging Face `pipeline("image-to-text")` |
| Story generation | Hugging Face `pipeline("text-generation")` |
| Speech | gTTS |
| Images | Pillow |
| Runtime | PyTorch, Python 3.10+ |

---

## 🤗 Models Used

| Model | Task |
|---|---|
| `Salesforce/blip-image-captioning-base` | Image → caption |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | Caption → story (primary) |
| `roneneldan/TinyStories-33M` | Caption → story (fallback) |

The primary story model is instruction-tuned, so it follows the "write a story about this scene" prompt. If it fails to load, the pipeline falls back to TinyStories, which is a completion model and cannot follow instructions — it only continues a seed sentence.

---

## 🔄 Pipeline

**1. Image → Caption.**
The uploaded image is validated (header and full decode), converted to RGB, and passed to BLIP. Beam search with a length penalty produces a richer caption than greedy decoding alone, e.g. *"a brown puppy playing in grass"*.

**2. Caption → Story.**
The caption's leading article is stripped, then substituted into a model-specific prompt. The primary model receives an instruction to write a 60-word story for a young child; the fallback receives a bare seed sentence. Output is cleaned (emoji/quote strip, ragged-ending repair) and trimmed to at most 100 words. Every word comes from the model — no hard-coded sentences are appended.

**3. Story → Audio.**
Emojis and quotes are stripped, then gTTS produces MP3 bytes rendered in an HTML5 audio player. If TTS fails, the story still renders so the user can read it.

---

## 🗂 Folder Structure

```
.
├── app.py                    # Main Streamlit application
├── constants.py              # Kid-safety blocklist + text-cleanup regexes
├── requirements.txt          # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml           # Theme + upload-size config
└── styles/
    ├── style.css             # Page layout and chrome
    ├── components.css        # Widget-level rules
    ├── animations.css        # All @keyframes
    └── responsive.css        # Media queries (loaded last)
```

The four CSS files are loaded in this order: `style.css → components.css → animations.css → responsive.css`. Later files override earlier ones on selector ties.

---

## 💻 Install & Run Locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. The first run downloads about 1.7 GB of models to `~/.cache/huggingface` — BLIP (~1 GB) and SmolLM2-360M (~720 MB). Later runs start in seconds.

---

## ☁️ Deploy on Streamlit Cloud

1. Push the project to a **public GitHub repository** — include `.streamlit/`, `styles/`, and `constants.py`.
2. Sign in at <https://share.streamlit.io>.
3. Click **New app**, select the repo, branch `main`, main file `app.py`.
4. No secrets needed — every model used is public on the Hugging Face Hub.
5. Click **Deploy**.

Streamlit Cloud installs from `requirements.txt`. A cold start takes 60–180 s while the model weights are downloaded into the container. Subsequent requests are fast.

Paste the resulting URL into the `Live app:` line at the top of this file once deployment is verified.

---

## 🧪 Testing

**Manual smoke test:**
Upload a photo with a clear subject → click **Make My Story!** → confirm a caption appears in the story flow → confirm the story is 50–100 words → confirm the audio player plays → confirm both download links work.

**Edge cases to check:**

- Corrupt or truncated image file → friendly "couldn't read that picture" message.
- Image over the upload limit → rejected by Streamlit before reaching the app.
- Offline mode → story still renders; a friendly message explains that audio is unavailable.
- Removing the upload mid-story → the story clears and the empty state returns.

---

## 🛡 Child-Safety Decisions

- Kid-safe primary model: SmolLM2-360M-Instruct is instruction-tuned and follows the "story for a young child" prompt; TinyStories (trained on a children's corpus) is the fallback.
- Word blocklist (violence, fear, substances, profanity, mature themes) in `constants.py` — checked against every generated story.
- Emojis and quotes are stripped before TTS so the voice reads naturally.
- Friendly error messages on every failure path — a traceback is never shown to the user.
- No storage, no accounts, no tracking. Uploaded images and generated stories live only in the current session's memory.