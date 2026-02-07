#!/bin/bash
# LLM Judge Scoring System - Example Run Script

# Configuration paths
INPUT_FILE="/mnt/llm-train/users/explore-train/wangzhenfang8/codes/generate/data/math500_and_gsm8k/math500_and_gsm8k_roll_8.jsonl"
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/dpo/yuqi_math_20k"
JUDGE_MODEL="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507"
SCRIPT_DIR="/mnt/llm-train/users/explore-train/qingyu/slimulation"

python $SCRIPT_DIR/recipe/llm_judge/prepare_judge.py \
    --input $INPUT_FILE \
    --output "$OUTPUT_DIR/prepare.jsonl" \
    --tokenizer $JUDGE_MODEL

python $SCRIPT_DIR/recipe/llm_judge/shard_jsonl.py \
    --input $OUTPUT_DIR/prepare.jsonl \
    --output-dir "$OUTPUT_DIR/shards_16_yuqi_new_data" \
    --num-shards 16 \
    --num-readers 48

# 1. 定义 Pod 列表 (按顺序排列，共16个)
pods=(
    # "dpo-data-0-6vd8m"
    # "dpo-data-1-qpcgs"
    # "dpo-data-2-sk8rp"
    "dpo-data-3-gj6wd"
    "dpo-data-4-hl59h"
    "dpo-data-5-rp6nz"
    "dpo-data-6-blg5d"
    "dpo-data-7-ch5qw"
    "dpo-data-8-8cgfz"
    "dpo-data-9-9f6vs"
    "dpo-data-10-q9vm6"
    "dpo-data-11-64mbg"
    "dpo-data-12-bprcz"
    "dpo-data-13-92rkn"
    "dpo-data-14-zm5dl"
    "dpo-data-15-j4hzj"
    "dpo-data-16-4fhn9"
    "dpo-data-17-gwkkg"
    "dpo-data-18-25k2x"
    "dpo-data-19-q92sh"
    "dpo-data-20-2rtp6"
    "dpo-data-21-6pmsb"
    "dpo-data-22-t45rt"
    "dpo-data-23-jg5m5"
)

# for pod in "${pods[@]}"; do echo "Cleaning $pod..."; kt exec $pod -- pkill -9 -f sglang || echo "No sglang process found on $pod"; done

# 2. 循环启动任务
for i in "${!pods[@]}"; do
    pod_name=${pods[$i]}
    shard_id=$i
    
    echo "正在为 Pod [$pod_name] 分配任务: Shard $shard_id ..."

    # 使用 nohup 或后台运行模式执行，避免 kubectl 断开导致任务终止
    kt exec $pod_name -- bash -c "pkill sglang && python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/inference.py \
        --input \"/mnt/llm-train/users/explore-train/qingyu/data/dpo/yuqi_math_20k/shards_16_yuqi_new_data/shard_${shard_id}.jsonl\" \
        --output \"/mnt/llm-train/users/explore-train/qingyu/data/dpo/yuqi_math_20k/responses/response_${shard_id}.jsonl\" \
        --model_path /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507 \
        --tp_size 1 \
        --dp_size 8 \
        --max_tokens 32768
    " > "log_shard_${shard_id}.log" 2>&1 &

    # 稍微等一秒，避免同时并发请求 kubectl API 过载
    sleep 1
done

python $SCRIPT_DIR/recipe/llm_judge/inference.py \
    --input "$OUTPUT_DIR/prepare.jsonl" \
    --output "$OUTPUT_DIR/response.jsonl" \
    --model_path $JUDGE_MODEL \
    --tp_size 1 \
    --dp_size 8 \
    --max_tokens 32768

python $SCRIPT_DIR/recipe/llm_judge/merge_and_extract.py \
    --response-dir $OUTPUT_DIR/responses \
    --output $OUTPUT_DIR/scores.jsonl \
    --failed $OUTPUT_DIR/failed.jsonl \
    --workers 48

python $SCRIPT_DIR/recipe/llm_judge/analyze_best_worst.py \
    --input $OUTPUT_DIR/scores.jsonl \
    --output $OUTPUT_DIR/final.jsonl

python $SCRIPT_DIR/recipe/llm_judge/analyze_scores.py \
    --scores $OUTPUT_DIR/final.jsonl

# python $SCRIPT_DIR/recipe/llm_judge/extract_scores.py \
#     --prepared $OUTPUT_DIR/prepare.jsonl \
#     --response $OUTPUT_DIR/response.jcsonl \
#     --output $OUTPUT_DIR/scores.jsonl \
#     --failed $OUTPUT_DIR/failed.jsonl

# # 1. 截取 100 条
# head -n 100 /mnt/llm-train/users/explore-train/qingyu/data/dpo/shards_16/shard_0.jsonl > shard_test_small.jsonl

# # 2. 运行推理测试
# python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/inference.py \
#     --input shard_test_small.jsonl \
#     --output shard_test_small_res.jsonl \
#     --model_path "/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507" \
#     --tp_size 2 \
#     --dp_size 4 \
#     --max_concurrency 128 \
#     --max_tokens 2048

# #!/bin/bash