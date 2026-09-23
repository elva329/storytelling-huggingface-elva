# 📖 StorySpark — A Storytelling App for Kids (ages 3–10)

> **Course:** ISOM5240 — Deep Learning Business Applications with Python
> **Assignment:** Individual Assignment
> **Author:** Elva

**Live app:** `https://storytelling-huggingface-elva.streamlit.app/`

StorySpark turns any picture into a short, 50–100-word bedtime story that the app then reads aloud. Built with Hugging Face Transformers and Streamlit, designed for ages 3–10.

## 🎯 Objective

Build a Python storytelling app that (1) accepts an uploaded image, (2) generates a 50–100-word narrative from it, (3) converts the story to audio, and (4) runs as an interactive Streamlit app deployable on Streamlit Cloud.

## ✨ Features

- Upload a picture (PNG / JPG / WEBP, ≤ 5 MB)
- Automatic image captioning (BLIP)
- 50–100-word story generation (TinyStories)
- Text-to-speech playback (gTTS)
- Download links for both story (`.txt`) and voice (`.mp3`)
- Kid-friendly UI: rainbow title, big buttons, animated loader, Ollie the Story Owl guide

## 🛠 Technologies

| Layer | Tool |
|---|---|
| UI | Streamlit + custom CSS |
| Captioning | Hugging Face `pipeline("image-to-text")` |
| Story generation | Hugging Face `pipeline("text-generation")` |
| Speech | gTTS |
| Images | Pillow |
| Runtime | PyTorch, Python 3.10+ |


## 🤗 Models Used

| Model | Task |
|---|---|
| `Salesforce/blip-image-captioning-base` | Image → caption |
| `roneneldan/TinyStories-33M` | Caption → story (primary) |
| `distilgpt2` | Caption → story (fallback) |


## 🔄 Pipeline

**1. Image → Caption.** Uploaded image is validated, converted to RGB, and passed to BLIP. Returns one line (e.g. *"a brown puppy playing in grass"*).

**2. Caption → Story.** The caption's leading article is stripped, a model-specific prompt is built, and the generator runs with sampling. Output is cleaned (emoji/quote strip, ragged-ending repair) and trimmed to ≤ 100 words. Every word comes from the model — no hard-coded sentences are appended.

**3. Story → Audio.** Emojis and quotes are stripped, then gTTS produces MP3 bytes rendered in an HTML5 player. If TTS fails, the story still shows.


## 🗂 Folder Structure

```
.
├── app.py
├── constants.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── styles/
    ├── style.css
    ├── components.css
    ├── animations.css
    └── responsive.css
```

## 💻 Install & Run Locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. First run downloads ~1.2 GB of models to `~/.cache/huggingface`; later runs start in seconds.


## ☁️ Deploy on Streamlit Cloud

1. Push the project to a **public GitHub repo** (include `.streamlit/` and `styles/`).
2. Sign in at <https://share.streamlit.io>.
3. Click **New app**, select the repo, branch `main`, main file `app.py`.
4. No secrets needed — all models are public.
5. Click **Deploy**.

Cold start takes 60–180 s while models download. Paste the resulting URL into the placeholder at the top of this file.


## 🧪 Testing

Manual smoke test: upload a photo with a clear subject → click **Make My Story!** → confirm caption, 50–100-word story, working audio player, and working download links.

Edge cases to check: corrupt image, oversized file, offline mode, removing the upload mid-story.


## 🛡 Child-Safety Decisions

- TinyStories-first: primary model was trained on a children's corpus.
- Word blocklist (violence, fear, substances, profanity, mature themes) in `constants.py`.
- Emojis/quotes stripped before TTS for natural reading.
- Friendly error messages — never a traceback.
- No storage, no accounts, no tracking.