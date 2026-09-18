import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_TOP_K
from app.services.rag_service import WonderlandRAGService


def evaluate_row(service, row):
    question = row["question"]
    expected_chapter = str(row["expected_chapter"]).strip()
    expected_status = row["expected_answer_status"]

    if service.is_blocked_input(question):
        retrieved_chunks = []
        retrieved_chapters = []
        top_distance = None
        retrieval_latency_ms = 0
        observed_status = "insufficient_evidence"
        input_blocked = True
    else:
        input_blocked = False
        start_time = time.perf_counter()

        retrieved_chunks = service.retrieve(
            question=question,
            top_k=DEFAULT_TOP_K,
        )

        retrieval_latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        has_evidence, top_distance = service.has_sufficient_evidence(
            retrieved_chunks
        )

        observed_status = (
            "grounded"
            if has_evidence
            else "insufficient_evidence"
        )

        retrieved_chapters = [
            chunk["metadata"]["chapter_number"]
            for chunk in retrieved_chunks
        ]

    hit_at_k = (
        expected_chapter in retrieved_chapters
        if expected_chapter
        else None
    )

    return {
        "test_id": row["test_id"],
        "category": row["category"],
        "question": question,
        "expected_chapter": expected_chapter or None,
        "retrieved_chapters": ", ".join(retrieved_chapters),
        "hit_at_k": hit_at_k,
        "expected_answer_status": expected_status,
        "observed_answer_status": observed_status,
        "status_match": observed_status == expected_status,
        "input_blocked": input_blocked,
        "top_retrieval_distance": top_distance,
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def main():
    input_path = (
        PROJECT_ROOT
        / "data"
        / "evaluation_questions.csv"
    )

    output_dir = PROJECT_ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "retrieval_evaluation_results.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {input_path}"
        )

    evaluation_df = pd.read_csv(
        input_path,
        keep_default_na=False,
    )

    service = WonderlandRAGService(
        groq_api_key="evaluation-only-placeholder",
    )

    results = [
        evaluate_row(service, row)
        for _, row in evaluation_df.iterrows()
    ]

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_path,
        index=False,
    )

    grounded_results = results_df[
        results_df["category"] == "grounded"
    ]

    retrieval_hit_rate = grounded_results["hit_at_k"].mean()
    status_accuracy = results_df["status_match"].mean()

    print("Evaluation complete.")
    print(f"Output file: {output_path}")
    print(
        f"Grounded retrieval Hit Rate@{DEFAULT_TOP_K}: "
        f"{retrieval_hit_rate:.0%}"
    )
    print(
        f"Guardrail status accuracy: "
        f"{status_accuracy:.0%}"
    )
    print()
    print(
        results_df[
            [
                "test_id",
                "category",
                "expected_chapter",
                "retrieved_chapters",
                "hit_at_k",
                "expected_answer_status",
                "observed_answer_status",
                "status_match",
                "input_blocked",
                "top_retrieval_distance",
                "retrieval_latency_ms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
