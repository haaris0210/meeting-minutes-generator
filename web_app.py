"""Run the local browser interface for the meeting-minutes project."""
from __future__ import annotations

import csv
import io
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import meeting_minutes as mm

HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 55 * 1024 * 1024
ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"


def train_minutes_maker():
    """Train the sentence model on the bundled labeled meeting dataset."""
    df = mm.load_data()
    dialogs = [mm.split_utterances(text) for text in df.dialogue]
    labels = [mm.oracle_labels(utterances, summary) for utterances, summary in zip(dialogs, df.summary)]
    y = np.concatenate(labels)
    features = mm.Features().fit(dialogs)
    X = features.transform(dialogs)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    classifier.fit(X, y)
    return mm.MinutesMaker(features, classifier, "Logistic Regression")


def review_questions(minutes: str) -> tuple[list[str], list[str]]:
    """Create transparent, transcript-grounded follow-up and challenge prompts."""
    lines = minutes.splitlines()

    def section(name, following):
        start = next((i for i, line in enumerate(lines) if line.strip().upper() == name), -1)
        if start < 0:
            return []
        end = len(lines) if not following else next((i for i in range(start + 1, len(lines)) if lines[i].strip().upper() == following), len(lines))
        return [line.strip().lstrip("-• ").strip() for line in lines[start + 1:end] if line.strip()]

    decisions = section("DECISIONS", "ACTION ITEMS")
    action_start = next((i for i, line in enumerate(lines) if line.strip().upper() == "ACTION ITEMS"), -1)
    action_lines = [line for line in lines[action_start + 1:] if line.strip()] if action_start >= 0 else []
    action_lines = [line for line in action_lines if not line.strip().lower().startswith("owner")]
    actions = []
    for line in action_lines:
        if len(line) >= 36:
            owner, due, task = line[2:14].strip(), line[14:36].strip(), line[36:].strip()
            if task:
                actions.append((owner, due, task))

    def decision_core(decision):
        text = decision.split(":", 1)[-1].strip()
        text = re.sub(r"^(?:Good idea[.!]?\s*|Agreed,?\s*|Okay, we decided to\s*|Okay,?\s*|Fine,?\s*|So it's settled,?\s*|Everyone agreed to\s*|We decided to\s*)", "", text, flags=re.I)
        text = re.sub(r"^(?:let's go with\s+|let's\s+|we will\s+|to\s+)", "", text, flags=re.I)
        text = text.rstrip(" .?!")[:100]
        if text.lower() in {"that is the decision then", "that's the decision then", "the decision then"}:
            return ""
        if text.lower().startswith("the top "):
            text = "use " + text
        return text

    questions = []
    for decision in decisions[:3]:
        if decision.lower() not in {"none found", "none detected"}:
            core = decision_core(decision)
            if core:
                questions.append(f"What evidence supports choosing to {core}?")
    for owner, due, task in actions[:3]:
        if re.match(r"^(?:sure[,!]?|okay[,!]?|yes[,!]?|will do|i'll take care of it|sounds good)", task, re.I):
            continue
        deadline = due or "the agreed date"
        if due and not due.lower().startswith(("by ", "before ", "on ", "until ")):
            deadline = f"by {due}"
        task_core = task[:100]
        date_match = mm.DATE_RE.search(task_core)
        if date_match and date_match.end() >= len(task_core) - 3:
            task_core = task_core[:date_match.start()].rstrip(" .")
        questions.append(f"What could block {owner or 'the owner'} from completing “{task_core}” {deadline}?")
    if not questions:
        questions = [
            "What important detail is still missing before the team can act?",
            "Who should own the next step, and when should progress be reviewed?",
        ]

    counterquestions = []
    for decision in decisions[:3]:
        if decision.lower() not in {"none found", "none detected"}:
            core = decision_core(decision)
            if core:
                counterquestions.append(f"What is the strongest case against choosing to {core}?")
    if actions:
        counterquestions.append("Which assumption could make these deadlines unrealistic, and what is the fallback?")
    if not counterquestions:
        counterquestions = [
            "What assumption in the discussion might be wrong?",
            "What alternative approach should the team compare before committing?",
        ]
    return list(dict.fromkeys(questions))[:5], list(dict.fromkeys(counterquestions))[:5]


class Handler(BaseHTTPRequestHandler):
    maker = None

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            raise OverflowError("That file is over the 50 MB limit. Please split it into smaller transcripts.")
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX_FILE.read_bytes())
        elif self.path == "/sample":
            sample = mm.SAMPLE_FILE.read_text(encoding="utf-8") if mm.SAMPLE_FILE.exists() else ""
            self._send(200, "text/plain; charset=utf-8", sample.encode("utf-8"))
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if self.path not in {"/summarize", "/scan"}:
            self.send_error(404, "Not found")
            return
        try:
            payload = self._read_json()
            if self.path == "/scan":
                filename = Path(str(payload.get("filename", "transcript.txt"))).name
                content = payload.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("The selected file is empty or unreadable.")
                if len(content.encode("utf-8")) > MAX_BODY_BYTES:
                    raise OverflowError("That file is over the 50 MB limit. Please split it into smaller transcripts.")
                suffix = Path(filename).suffix.lower()
                if suffix == ".csv":
                    rows = list(csv.DictReader(io.StringIO(content)))
                    if not rows:
                        raise ValueError("The CSV has no data rows.")
                    first = {str(key).strip().lower(): key for key in rows[0] if key is not None}
                    field = next((first[name] for name in ("dialogue", "transcript", "meeting_transcript", "conversation", "text") if name in first), None)
                    if field is None:
                        raise ValueError("CSV needs a dialogue, transcript, conversation, or text column.")
                    title_field = next((first[name] for name in ("meeting_id", "meeting", "title", "topic", "id") if name in first), None)
                    records = []
                    for index, row in enumerate(rows, start=1):
                        transcript = str(row.get(field, "")).strip()
                        if transcript:
                            label = str(row.get(title_field, "")).strip() if title_field else ""
                            records.append({"label": label or f"Meeting {index}", "transcript": transcript})
                    if not records:
                        raise ValueError("No transcript text was found in that CSV column.")
                    self._json(200, {"records": records, "count": len(records), "filename": filename})
                elif suffix in {".txt", ".md", ".log"}:
                    self._json(200, {"records": [{"label": filename, "transcript": content}], "count": 1, "filename": filename})
                else:
                    raise ValueError("Choose a UTF-8 .txt, .md, .log, or .csv file.")
                return

            transcript = payload.get("transcript", "")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError("Please enter a transcript.")
            minutes = self.maker.make(transcript)
            questions, counterquestions = review_questions(minutes)
            self._json(200, {"minutes": minutes, "model": self.maker.name,
                             "questions": questions, "counterquestions": counterquestions})
        except OverflowError as exc:
            self._json(413, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"Could not process the transcript: {exc}"})

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")


def main():
    print("Training the minutes model from the bundled dataset…")
    Handler.maker = train_minutes_maker()
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        active_port = PORT + 1
        server = ThreadingHTTPServer((HOST, active_port), Handler)
    else:
        active_port = PORT
    print(f"Meeting Minutes is ready: http://{HOST}:{active_port}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the local server…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
