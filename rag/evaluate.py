import argparse
import json

from rag.config import EVAL_QUESTIONS, REPORTS
from rag.retrieval import Retriever

KS = (1, 3, 5)


def evaluate_retrieval(retriever: Retriever, questions: list[dict], mode: str) -> dict:
    hits_at = {k: 0 for k in KS}
    reciprocal = 0.0
    misses = []
    for q in questions:
        results = retriever.search(q["question"], k=10, mode=mode)
        found = [i for i, h in enumerate(results) if h.chunk.law == q["law"] and h.chunk.article in q["articles"]]
        first = found[0] if found else None
        for k in KS:
            hits_at[k] += first is not None and first < k
        reciprocal += 1 / (first + 1) if first is not None else 0
        if first is None or first >= 5:
            misses.append(q["question"])
    n = len(questions)
    return {
        "mode": mode,
        **{f"recall@{k}": round(hits_at[k] / n, 3) for k in KS},
        "mrr@10": round(reciprocal / n, 3),
        "misses_top5": misses,
    }


def markdown(results: list[dict], n: int) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"{n} questions written in plain language, each with the article(s) that answer it.",
        "A hit means the right article appears in the top k results (one result per article).",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['mode']} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | {r['recall@5']:.2f} | {r['mrr@10']:.2f} |")
    best = max(results, key=lambda r: r["mrr@10"])
    if best["misses_top5"]:
        lines += ["", f"Questions outside the top 5 in `{best['mode']}` mode:", ""]
        lines += [f"- {q}" for q in best["misses_top5"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    args = parser.parse_args()

    questions = json.loads(EVAL_QUESTIONS.read_text(encoding="utf-8"))
    retriever = Retriever()
    results = [evaluate_retrieval(retriever, questions, mode) for mode in args.modes]
    for r in results:
        print(f"{r['mode']:<7} R@1 {r['recall@1']:.2f}  R@3 {r['recall@3']:.2f}  R@5 {r['recall@5']:.2f}  MRR {r['mrr@10']:.2f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retrieval_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "retrieval_eval.md").write_text(markdown(results, len(questions)), encoding="utf-8")


if __name__ == "__main__":
    main()
