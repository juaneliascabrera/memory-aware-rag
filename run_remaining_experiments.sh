#!/bin/bash

# Ejecutar K=4
uv run run_experiment.py \
    --k 4 \
    --granularity message \
    --use-reranker \
    --top-n 50 \
    --experiment-name bm25_message_rerank_top50_k4

# Ejecutar K=3 (se ejecutará automáticamente cuando termine el anterior)
uv run run_experiment.py \
    --k 3 \
    --granularity message \
    --use-reranker \
    --top-n 50 \
    --experiment-name bm25_message_rerank_top50_k3
