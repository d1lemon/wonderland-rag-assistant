import logging
import re
import time
from typing import Dict, List

import chromadb
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNKS_URL,
    DEFAULT_TOP_K,
    EDUCATIONAL_DISCLAIMER,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    INJECTION_BLOCKED_TERMS,
    INSUFFICIENT_EVIDENCE_ANSWER,
    MAX_RETRIEVAL_DISTANCE,
)


logger = logging.getLogger(__name__)


class WonderlandRAGService:
    def __init__(self, groq_api_key: str):
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
        self.groq_client = Groq(api_key=groq_api_key)

        self.chroma_client = chromadb.Client()
        self.collection_name = "wonderland_chunks_api"
        self.collection = self._build_collection()

    def _build_collection(self):
        chunks_df = pd.read_json(CHUNKS_URL)

        if chunks_df.empty:
            raise ValueError("The Alice chunk dataset is empty.")

        if not chunks_df["chunk_id"].is_unique:
            raise ValueError("Chunk IDs must be unique.")

        def make_embedding_text(row):
            return (
                f"Book: {row['book_title']}\n"
                f"Author: {row['book_author']}\n"
                f"Chapter {row['chapter_number']}: "
                f"{row['chapter_title']}\n\n"
                f"{row['chunk_text']}"
            )

        chunks_df["embedding_text"] = chunks_df.apply(
            make_embedding_text,
            axis=1,
        )

        embeddings = self.embedding_model.encode(
            chunks_df["embedding_text"].tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        try:
            self.chroma_client.delete_collection(
                self.collection_name
            )
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={
                "book_title": "Alice's Adventures in Wonderland",
                "corpus_version": "cleaned-v1",
                "embedding_model": EMBEDDING_MODEL_NAME,
                "distance_space": "l2",
            },
        )

        metadatas = []

        for _, row in chunks_df.iterrows():
            metadatas.append(
                {
                    "chapter_number": str(row["chapter_number"]),
                    "chapter_title": str(row["chapter_title"]),
                    "book_title": str(row["book_title"]),
                    "book_author": str(row["book_author"]),
                    "source_url": str(row["source_url"]),
                    "chunk_index": int(row["chunk_index"]),
                }
            )

        collection.add(
            ids=chunks_df["chunk_id"].tolist(),
            documents=chunks_df["chunk_text"].tolist(),
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        logger.info(
            "RAG collection built successfully with %s chunks.",
            collection.count(),
        )

        return collection

    def is_blocked_input(
        self,
        question: str,
    ) -> bool:
        normalized_question = question.lower()

        return any(
            blocked_term in normalized_question
            for blocked_term in INJECTION_BLOCKED_TERMS
        )

    def is_context_free_question(self, question: str) -> bool:
        normalized = re.sub(r"\s+", " ", question.casefold()).strip()
        normalized = normalized.rstrip("?.!").strip()
        return normalized in {
            "what happens next",
            "why does she do that",
        }

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:
        query_text = (
            "Book: Alice's Adventures in Wonderland\n"
            "Author: Lewis Carroll\n\n"
            f"Question: {question}"
        )

        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "chunk_id": results["ids"][0][index],
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": float(results["distances"][0][index]),
            }
            for index in range(len(results["ids"][0]))
        ]

    def has_sufficient_evidence(
        self,
        retrieved_chunks: List[Dict],
    ) -> tuple[bool, float | None]:
        if not retrieved_chunks:
            return False, None

        top_distance = retrieved_chunks[0]["distance"]

        return (
            top_distance <= MAX_RETRIEVAL_DISTANCE,
            top_distance,
        )

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        context_parts = []

        for chunk in retrieved_chunks:
            chunk_id = chunk["chunk_id"]
            chapter_number = chunk["metadata"]["chapter_number"]
            chunk_text = chunk["text"][:600]
            context_parts.append(
                f"[Chunk ID: {chunk_id} | Chapter {chapter_number}]\n"
                f"{chunk_text}"
            )

        context = "\n\n".join(context_parts)

        return f"""
    You are the Wonderland RAG Assistant.

    Use only the supplied source passages to answer the question.
    Do not use outside knowledge or invent details.
    Answer only when the supplied passages clearly support it.

    For every factual claim, cite one or more supporting source chunk IDs
    using exactly this format: [chunk-id].
    Use only chunk IDs shown in the supplied source passages.
    Do not use chapter citations, a citations heading, source labels, or
    any text inside citation brackets other than a chunk ID.

    End every answer with exactly one final line:
    "{EDUCATIONAL_DISCLAIMER}"

    Question:
    {question}

    Source passages:
    {context}
    """.strip()

    def validate_grounded_answer(
        self,
        answer: str,
        retrieved_chunks: List[Dict],
    ) -> bool:
        if not answer:
            return False

        if answer.count(EDUCATIONAL_DISCLAIMER) != 1:
            return False

        if not answer.endswith(EDUCATIONAL_DISCLAIMER):
            return False

        answer_body = answer.removesuffix(EDUCATIONAL_DISCLAIMER).rstrip()
        allowed_chunk_ids = {
            chunk["chunk_id"] for chunk in retrieved_chunks
        }
        citation_ids = re.findall(r"\[([^\[\]]+)\]", answer_body)

        if not citation_ids:
            return False

        return all(
            citation_id in allowed_chunk_ids
            for citation_id in citation_ids
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        prompt = self.build_prompt(question, retrieved_chunks)
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,
            )

            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            finish_reason = choice.finish_reason
            valid_contract = bool(content) and (
                finish_reason != "length"
                and self.validate_grounded_answer(content, retrieved_chunks)
            )

            if valid_contract:
                return content

            logger.warning(
                "Groq generation invalid | attempt=%s/%s | "
                "finish_reason=%s | content_empty=%s | contract_valid=%s",
                attempt,
                max_attempts,
                finish_reason,
                not bool(content),
                valid_contract,
            )

        raise RuntimeError(
            "Unable to generate a valid grounded answer after "
            f"{max_attempts} attempts."
        )

    def answer_question(
        self,
        question: str,
        top_k: int,
    ) -> Dict:
        total_start_time = time.perf_counter()

        if (
            self.is_blocked_input(question)
            or self.is_context_free_question(question)
        ):
            total_latency_ms = int(
                (time.perf_counter() - total_start_time) * 1000
            )

            logger.info(
                "Input declined before retrieval: override or missing context."
            )

            return {
                "answer": (
                    f"{INSUFFICIENT_EVIDENCE_ANSWER}\n\n"
                    f"{EDUCATIONAL_DISCLAIMER}"
                ),
                "retrieved_chunks": [],
                "latency_ms": total_latency_ms,
                "retrieval_latency_ms": 0,
                "generation_latency_ms": 0,
                "top_retrieval_distance": None,
                "answer_status": "insufficient_evidence",
            }

        retrieval_start_time = time.perf_counter()

        retrieved_chunks = self.retrieve(
            question=question,
            top_k=top_k,
        )

        retrieval_latency_ms = int(
            (time.perf_counter() - retrieval_start_time) * 1000
        )

        has_evidence, top_distance = self.has_sufficient_evidence(
            retrieved_chunks
        )

        if not has_evidence:
            total_latency_ms = int(
                (time.perf_counter() - total_start_time) * 1000
            )

            logger.info(
                "Weak retrieval evidence | top_distance=%s | "
                "threshold=%s | retrieval_latency_ms=%s",
                top_distance,
                MAX_RETRIEVAL_DISTANCE,
                retrieval_latency_ms,
            )

            return {
                "answer": (
                    f"{INSUFFICIENT_EVIDENCE_ANSWER}\n\n"
                    f"{EDUCATIONAL_DISCLAIMER}"
                ),
                "retrieved_chunks": retrieved_chunks,
                "latency_ms": total_latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": 0,
                "top_retrieval_distance": top_distance,
                "answer_status": "insufficient_evidence",
            }

        generation_start_time = time.perf_counter()

        try:
            answer = self.generate_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )
            answer_status = "grounded"
        except RuntimeError as error:
            logger.info(
                "Grounded-answer contract failed | reason=%s",
                error,
            )
            answer = (
                f"{INSUFFICIENT_EVIDENCE_ANSWER}\n\n"
                f"{EDUCATIONAL_DISCLAIMER}"
            )
            answer_status = "insufficient_evidence"

        generation_latency_ms = int(
            (time.perf_counter() - generation_start_time) * 1000
        )

        total_latency_ms = int(
            (time.perf_counter() - total_start_time) * 1000
        )

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "latency_ms": total_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "top_retrieval_distance": top_distance,
            "answer_status": answer_status,
        }
