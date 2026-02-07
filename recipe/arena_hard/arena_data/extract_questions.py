import argparse
import json
import re
import sys
import uuid
from typing import Iterable, List, Dict, Set


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"</?answer\b", re.IGNORECASE)
QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE)
QUESTION_TAG_RE = re.compile(r"</?question\b", re.IGNORECASE)


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_answer_block(text: str) -> List[str]:
    matches = [m.strip() for m in ANSWER_RE.findall(text)]
    matches = [m for m in matches if m]
    if not matches:
        return []
    if len(matches) != 1:
        return []
    block = matches[0]

    # If there are unmatched or stray <answer> tags, skip the whole response.
    stripped = ANSWER_RE.sub("", text)
    if ANSWER_TAG_RE.search(stripped):
        return []

    start = block.find("{")
    end = block.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    json_text = block[start : end + 1]
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict):
        return []

    items = []
    for key, value in payload.items():
        if not isinstance(key, str) or not key.startswith("question_"):
            continue
        if not isinstance(value, str):
            continue
        items.append((key, value.strip()))

    def _key_order(k: str) -> int:
        parts = k.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 10**9

    items.sort(key=lambda kv: _key_order(kv[0]))
    return [v for _, v in items if v]


def _extract_questions(text: str) -> List[str]:
    if not text:
        return []

    # Preferred format: <answer>{...}</answer>
    answer_questions = _parse_answer_block(text)
    if answer_questions:
        return _dedupe_keep_order(answer_questions)

    # Fallback: <question>...</question> tags
    matches = [m.strip() for m in QUESTION_RE.findall(text)]
    matches = [m for m in matches if m]
    if not matches:
        return []

    # If there are unmatched or stray <question> tags, skip the whole response.
    stripped = QUESTION_RE.sub("", text)
    if QUESTION_TAG_RE.search(stripped):
        return []
    return _dedupe_keep_order(matches)


def _strip_nested_question_prefix(question: str) -> str:
    parts = re.split(r"(?i)<question>", question)
    if len(parts) > 1:
        return parts[-1].strip()
    return question.strip()


def _new_uid(existing: Set[str]) -> str:
    while True:
        uid = uuid.uuid4().hex[:16]
        if uid not in existing:
            existing.add(uid)
            return uid


def extract_questions(input_file: str, output_file: str, max_per_seed: int) -> Dict[str, int]:
    total_seeds = 0
    total_questions = 0
    used_uids: Set[str] = set()

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            total_seeds += 1

            response = data.get("response", "")
            questions = _extract_questions(response)
            if max_per_seed > 0:
                questions = questions[:max_per_seed]

            for q in questions:
                q = _strip_nested_question_prefix(q)
                if not q:
                    continue
                record = {
                    "uid": _new_uid(used_uids),
                    "prompt": q,
                    "category": data.get("category", "hard_prompt"),
                    "subcategory": data.get("subcategory", "unknown"),
                    "seed_uid": data.get("uid"),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_questions += 1

    return {
        "seeds": total_seeds,
        "questions": total_questions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract questions from <answer>{...}</answer> or <question> tags."
    )
    parser.add_argument("--input", required=True, help="Input results.jsonl")
    parser.add_argument("--output", required=True, help="Output question.jsonl")
    parser.add_argument(
        "--max-per-seed",
        type=int,
        default=10,
        help="Max questions per seed (0 = no limit).",
    )
    args = parser.parse_args()

    stats = extract_questions(args.input, args.output, args.max_per_seed)
    print(
        f"[Extract] seeds={stats['seeds']}, questions={stats['questions']}, output={args.output}"
    )


if __name__ == "__main__":
    main()
