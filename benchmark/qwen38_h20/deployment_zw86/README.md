# Qwen3.8 Flash Next serving on 32 H20

Eight independent TP4/EP4 replicas, two per 8-GPU node, using BF16, NEXTN with three steps and four draft tokens, FlashInfer linear-attention prefill, Triton verification, 8192-token prefill chunks, and static memory fraction0.82. A round-robin SGLang router exposes the OpenAI endpoint.

The engine is based on main commit `a4a6eddd25343905245db7a8b54bcf24d9483c8b` plus the existing tested changes and the GPU JPEG RGB fix in `62d4f8941a0d36d009a09ae86bee6647dcb2b871`. The model is pinned to revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

Set `QWEN38_PROJECT` to the verified project root. The runtime sets PYTHONPATH to the project's independent source directory so a read-only reused venv cannot silently import another deployment's editable source.

Qualification includes per-worker deterministic chat, tools and history, thinking, complete SSE streams, PNG/RGB JPEG/grayscale JPEG, expanded vision usage counts, 32K retrieval, and a complete196604-token prompt with a65536-token output budget and four reserved MTP tokens. The next output token is rejected. Accepting that budget with a short generated answer does not test a65536-token generation. Synthetic fixtures do not replay completed evaluation questions.
