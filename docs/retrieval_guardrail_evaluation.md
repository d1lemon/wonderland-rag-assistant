# Retrieval Guardrail Evaluation

## Configuration

- Corpus: Alice's Adventures in Wonderland, cleaned chunk dataset
- Loaded chunk count: 197
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Vector store: Chroma in-memory collection
- Distance space: squared L2
- Threshold: 0.65
- Lower retrieval distance indicates a closer semantic match.

## Decision Rule

```text
top_retrieval_distance <= 0.65
    -> answer_status = grounded
    -> call Groq generation

top_retrieval_distance > 0.65
    -> answer_status = insufficient_evidence
    -> skip Groq generation
```

## Observed Tests

| Question | Top distance | Answer status | Generation latency | Outcome |
|---|---:|---|---:|---|
| Who does Alice follow down the rabbit-hole? | 0.3547 | grounded | 593–1139 ms | Cited Chapter I answer generated |
| What is the capital of Japan? | 0.7277 | insufficient_evidence | 0 ms | Controlled refusal; generation skipped |

## Conclusion

The initial threshold separated the tested supported question from the
tested out-of-corpus question. The zero generation latency for the
out-of-corpus question demonstrates that the backend made a
programmatic retrieval-confidence decision before calling the model.

The threshold remains corpus-specific and will be calibrated against a
larger labeled evaluation dataset.
