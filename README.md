# Fix 1 of 14 — Apply now

**File:** `README.md`
**Action:** Replace the entire file contents with the version below.

```markdown
# 📖 StorySpark — A Storytelling App for Kids (ages 3–10)

> ISOM5240 Deep Learning Business Applications with Python — Individual Assignment
> Author: Elva

**Live app:** https://storytelling-huggingface-elva.streamlit.app/

StorySpark turns any picture into a short, 50–100-word bedtime story
that the app then reads aloud. Built with Hugging Face Transformers
pipelines and a friendly Streamlit UI designed for little hands and
big imaginations. 🌈

---

## ✨ Features

| Step | What it does                  | Technology                                                   |
| ---- | ----------------------------- | ------------------------------------------------------------ |
| 1    | Understands an uploaded image | `Salesforce/blip-image-captioning-base` (image-to-text)      |
| 2    | Turns the caption into a tale | `roneneldan/TinyStories-33M` (fallback: `distilgpt2`)         |
| 3    | Reads the story out loud      | `gTTS` (Google Text-to-Speech → MP3)                          |
| 4    | Lets you save both outputs    | Two download links (voice MP3 + story TXT)                    |

Kid-friendly touches: big rounded buttons, a rainbow title, an
animated turning-book loader while the story is being written, and
Ollie the Story Owl as the empty-state guide.

---

## 🗂 Project layout

```
.
├── app.py                    # Main Streamlit application
├── constants.py              # Kid-safety blocklist + text-cleanup regexes
├── requirements.txt          # Python dependencies
├── runtime.txt               # Pins the Python version for Streamlit Cloud
├── .streamlit/
│   └── config.toml           # Theme + upload-size config
├── styles/
│   ├── style.css             # Page layout + chrome
│   ├── components.css        # Widget rules
│   ├── animations.css        # Every @keyframes
│   └── responsive.css        # Media queries (loaded last)
└── README.md
```

---

## 🚀 Run locally

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the two Hugging Face models (~1.2 GB total).
After that they're cached in `~/.cache/huggingface`, so subsequent
runs start in seconds. A warm-up step on the Streamlit side loads the
models before the first user uploads a picture.

Open <http://localhost:8501> and upload a picture to make your first story. 🎉

---

## ☁️ Deployment

This app targets **Streamlit Community Cloud**.

1. Push this folder to a public GitHub repository.
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **New app** → pick the repo, branch (`main`), and main file
   (`app.py`).
4. No secrets are required — every model used is public on the Hub.
5. Hit **Deploy**.

**Python version:** the app targets Python 3.11 (Streamlit Cloud's
current default). It requires Python 3.10+ because `app.py` uses
`X | Y` union syntax throughout.

---

## 🧠 Models at a glance

| Model                                  | Task              | Why we picked it |
| -------------------------------------- | ----------------- | ---------------- |
| `Salesforce/blip-image-captioning-base`| Image captioning  | Suggested in the assignment; compact and produces friendly captions |
| `roneneldan/TinyStories-33M`           | Story generation  | Trained on the TinyStories children's corpus — safe vocabulary |
| `distilgpt2` (fallback)                | Story generation  | Backup if TinyStories is unreachable |
| `gTTS`                                 | Text-to-speech    | No model to host; produces MP3 in a single call |

If `roneneldan/TinyStories-33M` is unavailable on the Hub, the app logs
the failure and silently switches to `distilgpt2`.

---

## 🛡 Safety notes for ages 3–10

* TinyStories emits short, plain-English prose using a child-safe
  vocabulary. `distilgpt2` is filtered through a blocklist before it
  is shown (`constants.py`).
* If a generation ever slips past the blocklist, the app retries up
  to two more times before surfacing a friendly message.
* No data is persisted: uploaded images are processed in memory and
  discarded when the session ends.

---

## 🧪 Quick smoke test

Upload any photo (a pet, a toy, a drawing) and click **Make My Story!**.
Within ~10–30 s you should see:

1. A short caption derived from the picture.
2. A 3–5-sentence story built around that subject.
3. An HTML5 audio player you can hit **▶** on, plus two download links.
