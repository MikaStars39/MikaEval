#!/bin/bash
# Complete evaluation pipeline example
# Usage: bash run.sh

# MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/sft-s1-lr1e-5-DECAY_SAMPLES-0212_1456"
MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/20260127_104615_opd_40b/iter_0000031_hf"
CACHE_DIR="/mnt/llm-train/users/explore-train/qingyu/.cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/eval_outputs/${TIMESTAMP}_opd_test_new_31"

# Step 1: Prepare data (load benchmarks and apply chat template)
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/prepare_data.py \
    --dataset "aime2024@8,aime2025@8,math500@4,minerva@4,amc2023@8,hmmt2025@8" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --prompt-format "lighteval"

# Step 2: Run batch inference
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/inference.py \
    --input "$OUTPUT_DIR/data.chat.jsonl" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --model "$MODEL_PATH" \
    --tp-size 1 \
    --dp-size 8 \
    --temperature 0.6 \
    --max-tokens 15000 \
    --resume

# Step 3: Evaluate and calculate metrics
python /mnt/llm-train/users/explore-train/qingyu/MikaEval/recipe/evaluation/evaluate.py \
    --input "$OUTPUT_DIR/results.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --num-proc 32
