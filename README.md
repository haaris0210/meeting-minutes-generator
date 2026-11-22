# Meeting Minutes Generator

A Statistical Machine Learning project: turn a meeting transcript into a **summary**, **decisions** and **action items**.

## Run it

```bash
pip install -r requirements.txt
python web_app.py
```

Then open the localhost URL printed in the terminal (normally **http://127.0.0.1:8000**; if that port is already in use, the app uses **http://127.0.0.1:8001**). The first startup trains the minutes model on the bundled dataset; leave the terminal window running while you use the app. Press `Ctrl+C` in the terminal to stop it.

The browser interface lets you paste or upload a transcript, load the sample, generate minutes, and copy the result. Upload UTF-8 `.txt`, `.md`, or `.log` transcripts, or a `.csv` dataset with a `dialogue`, `transcript`, `conversation`, or `text` column (up to 50 MB). CSV rows remain separate; select a meeting row to load and summarize it individually. The brief also proposes follow-up questions and counterquestions based on decisions and action items. Processing stays local. The web server uses Python’s standard library; no extra package is needed.

To run the original model comparison report and terminal demo instead, use:

```bash
python meeting_minutes.py
```
Use `python meeting_minutes.py --no-window` to skip the optional Tkinter window. No downloads, no internet, no API.

The script trains the models, prints the comparison tables, saves them to `results.txt`, shows example minutes, and can
open a small window (tkinter) where you paste any transcript. Use `python meeting_minutes.py --no-window` to skip the window.

## What it does

1. **Dataset** (`data/meetings.csv`): 300 generated meetings, each with a reference summary and the true positions of
   decisions and action items. It is created automatically if the file is missing. You can replace it with your own CSV
   with columns `dialogue, summary, action_idx, decision_idx`.
2. **Important-sentence classifiers**: Logistic Regression, Linear SVM and Random Forest. Features are TF-IDF, sentence
   position, length, digits, question mark, cue words ("will", "agreed", "please") and similarity to the whole meeting.
   Training labels come from a ROUGE oracle: sentences that best match the reference summary are labeled 1.
3. **Baselines**: first 3 sentences, and TextRank (PageRank over a sentence-similarity graph).
4. **Action items and decisions**: rule-based detection of owner, task and due date, compared with a TF-IDF + Logistic
   Regression detector.
5. **Evaluation**: ROUGE-1/2/L (implemented in the file), precision, recall and F1 on a held-out 20% of meetings.

## Files

```
meeting_minutes.py   everything: data, models, evaluation, minutes, window
data/meetings.csv    dataset
data/sample_meeting.txt  example transcript
requirements.txt
```

## Limitations

- The dataset is synthetic (template based), so scores are higher than they would be on real meetings, and the
  TF-IDF detectors reaching 1.0 mostly shows the templates are easy.
- Summaries are extractive (copied sentences), not rewritten.
- Rules miss unusual phrasings. Real transcripts (AMI, QMSum) would be the natural next step.
