"""
Meeting Minutes Generator  (Statistical Machine Learning project)
=================================================================
Takes a meeting transcript and produces a summary, decisions and action items.

Just run this file (F5 in IDLE, or `python meeting_minutes.py`). It will:
  1. load the dataset  (data/meetings.csv, created automatically if missing)
  2. train three ML models that pick the important sentences of a meeting
  3. compare them with two baselines using ROUGE, precision, recall and F1
  4. print + save the results to results.txt
  5. show example minutes and optionally open a small window where you can paste any transcript

Only needs: numpy, pandas, scikit-learn  (tkinter ships with Python).
"""
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "meetings.csv"
SAMPLE_FILE = ROOT / "data" / "sample_meeting.txt"
K = 3  # number of sentences used in an extractive summary


# =============================================================================
# 1. DATASET  (synthetic meetings with known summaries, decisions and action items)
# =============================================================================
NAMES = ["Aman", "Riya", "Kabir", "Zoya", "Priya", "Rahul", "Neha", "Arjun", "Sara", "Dev", "Meera", "Ishaan"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "next week", "the end of the week"]
GREETINGS = ["Good morning everyone, let's get started.", "Hi all, thanks for joining the meeting.",
             "Okay, let's begin the weekly sync.", "Hello team, shall we start?"]
FILLERS = ["How was everyone's weekend?", "The coffee machine is broken again.",
           "I will be late tomorrow because of the train strike.", "Please keep your phones on silent.",
           "The office is very cold today.", "Did anyone watch the match yesterday?",
           "I think the last meeting was too long.", "We should really fix the air conditioning.",
           "Nice weather today, isn't it?", "Sorry, my internet is a bit slow.",
           "The last release went out on Friday.", "Someone should really fix the air conditioning.",
           "We decided to skip the office party last year.", "Please ignore my background noise."]
QUESTIONS = ["How serious is it?", "What do you all think we should do?", "Has this happened before?",
             "Is there any quick fix for that?"]
CLOSINGS = ["Thanks everyone, that's all for today.", "Great, let's end the meeting here.",
            "Alright, see you all next week."]
# (problem, decision, task 1, task 2)
SCENARIOS = [
    ("the login page is loading slowly", "optimize the database queries", "rewrite the slow login queries", "run a load test on the login page"),
    ("the payment module is delayed", "ask the bank for faster API access", "email the bank about the API keys", "update the project timeline"),
    ("the weekly report has many errors", "automate the report generation", "write the report automation script", "check last month's numbers"),
    ("customers complain about slow delivery", "switch back to the old courier", "inform the warehouse manager", "update the delivery policy"),
    ("the mobile app crashes on startup", "release a hotfix this week", "fix the startup crash", "test the hotfix on Android"),
    ("our cloud bill increased by thirty percent", "shut down the unused test servers", "set up automatic server shutdown", "review the storage plan"),
    ("the new hires need more training", "run two onboarding workshops", "book the training room", "prepare the workshop slides"),
    ("the website homepage looks outdated", "redesign the homepage", "prepare new homepage mockups", "collect feedback from the sales team"),
    ("the nightly data pipeline keeps failing", "add automatic retries", "implement the retry logic", "add an alert to the team channel"),
    ("the security audit found weak passwords", "enforce a stronger password policy", "update the password rules", "send a notice to all staff"),
    ("the team meetings run too long", "limit meetings to thirty minutes", "send out a new meeting schedule", "create a shared agenda template"),
    ("the annual meetup still has no venue", "book the community hall", "confirm the hall booking", "order snacks for the event"),
]


def make_meeting(rng):
    a, b, d, e = rng.sample(NAMES, 4)
    problem, decision, task1, task2 = rng.choice(SCENARIOS)
    day1, day2 = rng.choice(DAYS), rng.choice(DAYS)
    utts, actions, decisions = [], [], []

    def add(spk, text, kind=None):
        utts.append((spk, text))
        if kind == "action":
            actions.append(len(utts) - 1)
        if kind == "decision":
            decisions.append(len(utts) - 1)

    add(a, rng.choice(GREETINGS))
    for _ in range(rng.randint(0, 2)):
        add(rng.choice([b, d, e]), rng.choice(FILLERS))
    add(b, rng.choice(["I noticed that {}.", "Just so you know, {}.", "The main issue is that {}."]).format(problem))
    add(a, rng.choice(QUESTIONS))
    add(d, f"We could {decision}.")
    add(a, rng.choice(["Agreed, let's {}.", "Okay, we decided to {}.", "Fine, let's {}.",
                    "Everyone agreed to {}.", "So it's settled, we will {}."]).format(decision), "decision")

    form = rng.choice(["self", "self2", "ask", "name", "going"])
    if form == "self":
        add(d, f"I will {task1} by {day1}.", "action")
    elif form == "self2":
        add(d, f"Let me {task1} by {day1}.", "action")
    elif form == "going":
        add(a, f"{d} is going to {task1} by {day1}.", "action")
    elif form == "ask":
        add(a, f"{d}, can you {task1} by {day1}?", "action")
        add(d, rng.choice(["Sure, I'll take care of it.", "Yes, no problem.", "Okay, will do."]))
    else:
        add(a, f"{d} will {task1} by {day1}.", "action")
    summary = f"{b} reported that {problem}. The team decided to {decision}. {d} will {task1} by {day1}."

    if rng.random() < 0.7:
        if rng.random() < 0.5:
            add(a, f"{e}, please {task2} before {day2}.", "action")
        else:
            add(a, f"{e} should {task2} by {day2}.", "action")
        summary += f" {e} will {task2} by {day2}."
    for _ in range(rng.randint(0, 2)):
        add(rng.choice([b, d, e]), rng.choice(FILLERS))
    add(a, rng.choice(CLOSINGS))

    return {"dialogue": "\n".join(f"{s}: {t}" for s, t in utts), "summary": summary,
            "action_idx": ";".join(map(str, actions)), "decision_idx": ";".join(map(str, decisions))}


def load_data(n=300, seed=42):
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(exist_ok=True)
        rng = random.Random(seed)
        pd.DataFrame([make_meeting(rng) for _ in range(n)]).to_csv(DATA_FILE, index=False)
        print(f"[data] Created {DATA_FILE.name} with {n} meetings")
    return pd.read_csv(DATA_FILE, keep_default_na=False)


def split_utterances(text):
    """'Name: sentence' lines -> list of (speaker, text)."""
    out = []
    for line in str(text).strip().splitlines():
        line = line.strip()
        if line:
            m = re.match(r"^([A-Za-z][\w .'-]{0,30}?):\s*(.+)$", line)
            out.append((m.group(1).strip(), m.group(2).strip()) if m else ("Unknown", line))
    return out


def parse_idx(s):
    return {int(x) for x in str(s).split(";") if x != ""}


# =============================================================================
# 2. ROUGE (implemented here so no extra library is needed)
# =============================================================================
def tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _f1(overlap, n_pred, n_ref):
    if overlap == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_n(ref, pred, n):
    def grams(toks):
        return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    r, p = grams(tokens(ref)), grams(tokens(pred))
    return _f1(sum((r & p).values()), max(sum(p.values()), 1), max(sum(r.values()), 1))


def rouge_l(ref, pred):
    a, b = tokens(ref), tokens(pred)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return _f1(dp[-1][-1], max(len(b), 1), max(len(a), 1))


# =============================================================================
# 3. MACHINE LEARNING: which sentences belong in the summary?
# =============================================================================
def oracle_labels(utts, reference, max_select=3):
    """Label 1 for the sentences that best match the human summary (greedy ROUGE-1)."""
    texts = [t for _, t in utts]
    chosen, best = [], 0.0
    while len(chosen) < max_select:
        cand = None
        for i in range(len(texts)):
            if i not in chosen:
                score = rouge_n(reference, " ".join(texts[j] for j in sorted(chosen + [i])), 1)
                if score > best:
                    best, cand = score, i
        if cand is None:
            break
        chosen.append(cand)
    return [int(i in chosen) for i in range(len(texts))]


CUES = ["will", "decid", "agree", "let's", "please", "should", "can you", "by ", "need", "tomorrow"]


class Features:
    """TF-IDF of each sentence + position, length, digits, question mark, cue words, similarity to the meeting."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        self.scaler = StandardScaler()

    def _dense(self, dialogs, X):
        rows, start = [], 0
        for d in dialogs:
            n = len(d)
            Xd, start = X[start:start + n], start + n
            centroid = np.asarray(Xd.mean(axis=0)).ravel()
            sim = Xd.dot(centroid) / (np.linalg.norm(centroid) + 1e-9)
            for i, (_, t) in enumerate(d):
                low = t.lower()
                rows.append([i / max(n - 1, 1), i == 0, i == n - 1, len(t.split()), bool(re.search(r"\d", t)),
                             "?" in t, sum(c in low for c in CUES), sim[i]])
        return np.array(rows, dtype=float)

    def fit(self, dialogs):
        texts = [t for d in dialogs for _, t in d]
        self.tfidf.fit(texts)
        self.scaler.fit(self._dense(dialogs, self.tfidf.transform(texts)))
        return self

    def transform(self, dialogs):
        X = self.tfidf.transform([t for d in dialogs for _, t in d])
        return sparse.hstack([X, sparse.csr_matrix(self.scaler.transform(self._dense(dialogs, X)))]).tocsr()


def new_models():
    return {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Linear SVM": LinearSVC(class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)}


def model_scores(clf, X):
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)


def textrank_scores(utts, damping=0.85, iters=50):
    """Baseline without any training: PageRank over a sentence-similarity graph."""
    texts = [t for _, t in utts]
    n = len(texts)
    try:
        S = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))
    except ValueError:
        return np.ones(n)
    np.fill_diagonal(S, 0)
    sums = S.sum(axis=1, keepdims=True)
    T = np.where(sums > 0, S / np.where(sums == 0, 1, sums), 1.0 / n)
    r = np.ones(n) / n
    for _ in range(iters):
        r = (1 - damping) / n + damping * T.T @ r
    return r


def top_k(scores, k=K):
    return sorted(np.argsort(-np.asarray(scores))[:min(k, len(scores))].tolist())


# =============================================================================
# 4. ACTION ITEMS + DECISIONS  (rules; compared with an ML detector in the report)
# =============================================================================
ACTION_RE = re.compile(r"\bI(?:'ll|\s+will|\s+can|\s+am going to|\s+have to|\s+need to)\b|\b(?:can|could|would|will) you\b"
                       r"|\bplease\b|\bneeds? to\b|\b(?:should|must|have to|has to)\b|\blet me\b", re.I)
NAME_WILL = re.compile(r"\b([A-Z][a-z]{2,})\s+(?:will|should|needs to|has to|can|is going to)\b")
NOT_NAMES = {"We", "They", "That", "This", "There", "What", "Who", "How", "You", "And", "But", "Then", "The", "It"}
DECISION_RE = re.compile(r"\bagreed\b|\bdecided\b|\bdecision\b|\bapproved\b|let's go with"
                         r"|let's (?:book|invite|put|add|switch|return|ship|launch|move forward)", re.I)
DAYS_RE = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_RE = re.compile(rf"\b(?:by |before |on |until )?(today|tonight|tomorrow|end of (?:the )?(?:day|week|month)|next week|"
                     rf"(?:{DAYS_RE})|\d{{1,2}}(?:st|nd|rd|th)? (?:of )?[A-Za-z]+)\b", re.I)


def is_action(text):
    m = NAME_WILL.search(text)
    return bool(ACTION_RE.search(text) or (m and m.group(1) not in NOT_NAMES))


def is_decision(text):
    return bool(DECISION_RE.search(text))


def find_owner(speaker, text, speakers):
    if re.search(r"\bI(?:'ll|\s+will|\s+can|\s+am|\s+have|\s+need)\b|\blet me\b", text):
        return speaker
    m = re.match(r"^\s*([A-Z][a-z]+),?\s+(?:(?:can|could|would|will)\s+you|please)\b", text) or NAME_WILL.search(text)
    if m and m.group(1) in speakers:
        return m.group(1)
    return "Team" if re.search(r"\bwe\b", text, re.I) else "Unassigned"


# =============================================================================
# 5. MINUTES
# =============================================================================
class MinutesMaker:
    def __init__(self, features, clf, name):
        self.features, self.clf, self.name = features, clf, name

    def make(self, transcript, k=K):
        utts = split_utterances(transcript)
        if not utts:
            return "Please enter a transcript (one line per utterance, 'Name: text')."
        idx = top_k(model_scores(self.clf, self.features.transform([utts])), k)
        lines = ["MEETING MINUTES", f"(summary model: {self.name})", "", "SUMMARY"]
        lines += [f"  {utts[i][1]}" for i in idx]
        lines += ["", "DECISIONS"]
        lines += [f"  - {s}: {t}" for s, t in utts if is_decision(t)] or ["  - none found"]
        lines += ["", "ACTION ITEMS", f"  {'Owner':<12}{'Due':<22}Task"]
        found, speakers = False, {s for s, _ in utts}
        for s, t in utts:
            if is_action(t):
                found = True
                due = DATE_RE.search(t)
                lines.append(f"  {find_owner(s, t, speakers):<12}{(due.group(0).strip() if due else '-'):<22}{t}")
        if not found:
            lines.append("  - none found")
        return "\n".join(lines)


# =============================================================================
# 6. SMALL WINDOW (tkinter)
# =============================================================================
def open_window(maker):
    import tkinter as tk
    from tkinter import scrolledtext
    root = tk.Tk()
    root.title("Meeting Minutes Generator")
    tk.Label(root, text="Paste a transcript (one line per utterance, 'Name: text'):").pack(anchor="w", padx=8, pady=4)
    box = scrolledtext.ScrolledText(root, width=95, height=14)
    box.pack(padx=8)
    if SAMPLE_FILE.exists():
        box.insert("1.0", SAMPLE_FILE.read_text(encoding="utf-8"))
    out = scrolledtext.ScrolledText(root, width=95, height=18, state="disabled")

    def generate():
        out.config(state="normal")
        out.delete("1.0", "end")
        out.insert("1.0", maker.make(box.get("1.0", "end")))
        out.config(state="disabled")

    tk.Button(root, text="Generate minutes", command=generate).pack(pady=6)
    out.pack(padx=8, pady=(0, 8))
    root.mainloop()


# =============================================================================
# 7. MAIN: train, evaluate, demo
# =============================================================================
def table(rows):
    cols = list(rows[0])
    w = [max(len(str(c)), *(len(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols]
    line = lambda vals: "  ".join(str(v).ljust(x) for v, x in zip(vals, w))
    out = [line(cols), line(["-" * x for x in w])]
    out += [line([f"{r[c]:.3f}" if isinstance(r[c], float) else r[c] for c in cols]) for r in rows]
    return "\n".join(out)


def prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"Precision": p, "Recall": r, "F1": f}


def main():
    report = []

    def show(text=""):
        print(text)
        report.append(text)

    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    def prepare(frame):
        dialogs = [split_utterances(d) for d in frame.dialogue]
        labels = [oracle_labels(u, s) for u, s in zip(dialogs, frame.summary)]
        return dialogs, labels

    tr_dialogs, tr_labels = prepare(train_df)
    te_dialogs, te_labels = prepare(test_df)
    y_tr, y_te = np.concatenate(tr_labels), np.concatenate(te_labels)
    show(f"Dataset: {len(train_df)} training meetings, {len(test_df)} test meetings "
         f"({len(y_tr)} / {len(y_te)} sentences, {y_tr.mean():.0%} are 'important')")

    feats = Features().fit(tr_dialogs)
    X_tr, X_te = feats.transform(tr_dialogs), feats.transform(te_dialogs)
    models = new_models()
    for clf in models.values():
        clf.fit(X_tr, y_tr)

    # ---- A. which sentences are important? --------------------------------
    show("\nA) Sentence importance classifiers (test set, labels from the ROUGE oracle)")
    rows = [{"Model": name, **prf(y_te, clf.predict(X_te))} for name, clf in models.items()]
    show(table(rows))

    # ---- B. summary quality (ROUGE) ----------------------------------------
    def pick(d, idx):
        return " ".join(d[i][1] for i in idx)

    preds = {f"Baseline: first {K} sentences": [pick(d, range(min(K, len(d)))) for d in te_dialogs],
             "Baseline: TextRank": [pick(d, top_k(textrank_scores(d))) for d in te_dialogs]}
    for name, clf in models.items():
        scores, out, start = model_scores(clf, X_te), [], 0
        for d in te_dialogs:
            out.append(pick(d, top_k(scores[start:start + len(d)])))
            start += len(d)
        preds[name] = out
    preds["Upper bound (oracle)"] = [pick(d, [i for i, v in enumerate(l) if v]) for d, l in zip(te_dialogs, te_labels)]
    refs = list(test_df.summary)
    rows = [{"Method": name, "ROUGE-1": np.mean([rouge_n(r, p, 1) for r, p in zip(refs, ps)]),
             "ROUGE-2": np.mean([rouge_n(r, p, 2) for r, p in zip(refs, ps)]),
             "ROUGE-L": np.mean([rouge_l(r, p) for r, p in zip(refs, ps)])} for name, ps in preds.items()]
    show(f"\nB) Summary quality on {len(refs)} test meetings (ROUGE F1, higher is better)")
    show(table(rows))
    best = max((r for r in rows if r["Method"] in models), key=lambda r: r["ROUGE-1"])["Method"]
    show(f"\nBest model for the minutes: {best}")

    # ---- C. action items and decisions: rules vs machine learning ----------
    def utterance_labels(frame, col):
        y = []
        for d, idx in zip(frame.dialogue, frame[col]):
            n, wanted = len(split_utterances(d)), parse_idx(idx)
            y += [int(i in wanted) for i in range(n)]
        return np.array(y)

    flat = lambda dialogs: [t for d in dialogs for _, t in d]
    rows = []
    for title, col, rule in [("Action items", "action_idx", is_action), ("Decisions", "decision_idx", is_decision)]:
        ya_tr, ya_te = utterance_labels(train_df, col), utterance_labels(test_df, col)
        rows.append({"Task": title, "Detector": "Rules", **prf(ya_te, [int(rule(t)) for t in flat(te_dialogs)])})
        vec = TfidfVectorizer(ngram_range=(1, 2))
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vec.fit_transform(flat(tr_dialogs)), ya_tr)
        rows.append({"Task": title, "Detector": "TF-IDF + LogReg",
                     **prf(ya_te, lr.predict(vec.transform(flat(te_dialogs))))})
    show("\nC) Detecting action items and decisions (sentence level, test set)")
    show(table(rows))

    # ---- D. example ---------------------------------------------------------
    maker = MinutesMaker(feats, models[best], best)
    sample = SAMPLE_FILE.read_text(encoding="utf-8") if SAMPLE_FILE.exists() else test_df.dialogue.iloc[0]
    show("\n" + "=" * 70 + "\nEXAMPLE\n" + "=" * 70)
    show(maker.make(sample))
    (ROOT / "results.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n[saved] results.txt")

    if "--no-window" not in sys.argv and input("\nOpen the transcript window? (y/n): ").strip().lower() == "y":
        try:
            open_window(maker)
        except Exception as e:
            print(f"Could not open the window ({e}). The results above are still valid.")


if __name__ == "__main__":
    main()
