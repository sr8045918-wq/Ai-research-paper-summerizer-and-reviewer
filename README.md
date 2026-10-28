# 📄 PaperLens: AI Research Paper Summarizer & Literature Reviewer

Upload research papers (PDF) or paste text, and get clear summaries, plain-language explanations,
paper Q&A, and multi-paper literature reviews. Built with **Streamlit** and **Google Gemini Flash**.

## Features
- **Structured summary**: problem, contributions, methodology, results, limitations, future work
- **Explanation level**: Beginner / Intermediate / Expert
- **Output language**: English, Urdu, Hindi, Arabic, and more
- **Extra tools**: ELI5, glossary, critical review, flashcards, search keywords, slide outline
- **Chat with the paper**: ask follow-up questions
- **Literature review**: upload 2-10 papers and get a comparison table, synthesis, gaps, and future directions
- **Download** notes and reviews as Markdown

## Run locally
```bash
git clone https://github.com/<your-username>/paperlens.git
cd paperlens
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey. You can paste it in the app sidebar,
or set it once:

- **Environment variable:** `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`)
- **Secrets file:** create `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your-key"`
  (**never commit this file**)

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add: `GEMINI_API_KEY = "your-key"`
5. Click **Deploy**.

## Notes & limitations
- Works with text-based PDFs. Scanned image PDFs have no extractable text (OCR is not included).
- Very long papers are truncated to fit the model's context window.
- AI can make mistakes. Verify important claims against the original paper.
- Model name is configurable in the sidebar (default: `gemini-flash-latest`).

## Tech stack
Streamlit · google-genai · pypdf
