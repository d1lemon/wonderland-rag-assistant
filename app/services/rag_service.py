import logging
import time
from typing import Dict, List

import chromadb
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNKS_URL,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
)


logger = logging.getLogger(__name__)


class WonderlandRAGService:
    def __init__(self, groq_api_key: str):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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
                f"Chapter {row['chapter_number']}: {row['chapter_title']}\n\n"
                f"{row['chunk_text']}"
            )

        chunks_df["embedding_text"] = chunks_df.apply(
            make_embedding_text,
            axis=1
        )

        embeddings = self.embedding_model.encode(
            chunks_df["embedding_text"].tolist(),
            show_progress_bar=False,
            normalize_embeddings=True
        )

        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={
                "book_title": "Alice's Adventures in Wonderland",
                "corpus_version": "cleaned-v1",
                "embedding_model": EMBEDDING_MODEL_NAME
            }
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
                    "chunk_index": int(row["chunk_index"])
                }
            )

        collection.add(
            ids=chunks_df["chunk_id"].tolist(),
            documents=chunks_df["chunk_text"].tolist(),
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        logger.info(
            "RAG collection built successfully with %s chunks.",
            collection.count()
        )

        return collection

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        query_text = (
            "Book: Alice's Adventures in Wonderland\n"
            "Author: Lewis Carroll\n\n"
            f"Question: {question}"
        )

        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        for index in range(len(results["ids"][0])):
            retrieved_chunks.append(
                {
                    "chunk_id": results["ids"][0][index],
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": float(results["distances"][0][index])
                }
            )

        return retrieved_chunks

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        context_parts = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            chapter_number = chunk["metadata"]["chapter_number"]
            chunk_text = chunk["text"][:600]

            context_parts.append(
                f"[Source {index}: Chapter {chapter_number}]\n{chunk_text}"
            )

        context = "\n\n".join(context_parts)

        return f"""
You are the Wonderland RAG Assistant.

Use only the supplied source passages to answer the question.
Do not use outside knowledge or invent details.

If the passages do not clearly support an answer, respond exactly:
"I do not have sufficient support in the provided source material to answer that."

After each factual claim, cite its chapter as [Chapter X].

End every answer with exactly:
"Educational information only—not personalized advice."

Question:
{question}

Source passages:
{context}
""".strip()

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        prompt = self.build_prompt(question, retrieved_chunks)

        response = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=180,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    def answer_question(
        self,
        question: str,
        top_k: int
    ) -> Dict:
        start_time = time.perf_counter()

        retrieved_chunks = self.retrieve(
            question=question,
            top_k=top_k
        )

        answer = self.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "latency_ms": latency_ms
        }
