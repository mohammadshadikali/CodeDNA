"""Read commits.csv, extract features per commit, and save to features.csv.

Expected input CSV columns: commit_hash, author_name, author_email, date, message, files_changed

Output CSV columns include: commit_hash, label (author_name), commit_hour, commit_day_of_week,
message_length, avg_line_length, snake_case_ratio, camelCase_ratio, comment_ratio, files_changed_count
"""
import argparse
import csv
import re
from datetime import datetime
from typing import List


SNAKE_RE = re.compile(r"^[a-z]+(?:_[a-z0-9]+)+$")
CAMEL_RE = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]+)+$")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def parse_iso_datetime(s: str) -> datetime:
    if not s:
        raise ValueError("Empty date string")
    try:
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        # Fallback: try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        raise


def tokenize_message(msg: str) -> List[str]:
    return TOKEN_RE.findall(msg or "")


def is_snake(token: str) -> bool:
    return bool(SNAKE_RE.match(token))


def is_camel(token: str) -> bool:
    return bool(CAMEL_RE.match(token))


def avg_line_length(msg: str) -> float:
    lines = (msg or "").splitlines()
    if not lines:
        return 0.0
    return sum(len(l) for l in lines) / len(lines)


def comment_ratio(msg: str) -> float:
    lines = (msg or "").splitlines()
    if not lines:
        return 0.0
    count = 0
    for l in lines:
        s = l.lstrip()
        if not s:
            continue
        if s.startswith(("#", "//", "/*", "*")) or "/*" in s or "*/" in s:
            count += 1
    return count / len(lines)


def files_changed_count(files_changed: str) -> int:
    if not files_changed:
        return 0
    parts = [p for p in files_changed.split(";") if p.strip()]
    return len(parts)


def extract_features_from_row(row: dict) -> dict:
    message = row.get("message", "") or ""
    tokens = tokenize_message(message)
    total_tokens = len(tokens) or 1
    snake = sum(1 for t in tokens if is_snake(t))
    camel = sum(1 for t in tokens if is_camel(t))

    date_str = row.get("date", "")
    try:
        dt = parse_iso_datetime(date_str)
        hour = dt.hour
        weekday = dt.weekday()
    except Exception:
        hour = ""
        weekday = ""

    return {
        "commit_hash": row.get("commit_hash", ""),
        "label": row.get("author_name", ""),
        "commit_hour": hour,
        "commit_day_of_week": weekday,
        "message_length": len(message),
        "avg_line_length": round(avg_line_length(message), 3),
        "snake_case_ratio": round(snake / total_tokens, 6),
        "camelCase_ratio": round(camel / total_tokens, 6),
        "comment_ratio": round(comment_ratio(message), 6),
        "files_changed_count": files_changed_count(row.get("files_changed", "")),
    }


def process(input_csv: str, output_csv: str):
    with open(input_csv, newline='', encoding='utf-8') as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)

    features = [extract_features_from_row(r) for r in rows]

    fieldnames = [
        "commit_hash",
        "label",
        "commit_hour",
        "commit_day_of_week",
        "message_length",
        "avg_line_length",
        "snake_case_ratio",
        "camelCase_ratio",
        "comment_ratio",
        "files_changed_count",
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for f in features:
            writer.writerow(f)


def parse_args():
    p = argparse.ArgumentParser(description="Extract features from commits.csv and save to features.csv")
    p.add_argument('--input', '-i', default='commits.csv', help='Input commits CSV file')
    p.add_argument('--output', '-o', default='features.csv', help='Output features CSV file')
    return p.parse_args()


def main():
    args = parse_args()
    process(args.input, args.output)
    print(f"Wrote features to {args.output}")


if __name__ == '__main__':
    main()
# Read commits.csv and extract developer style features for each commit
# Features: snake_case ratio, camelCase ratio, avg line length, comment ratio,
# commit hour, commit day of week, message length, files changed count
# Save output to features.csv with author_name as label column