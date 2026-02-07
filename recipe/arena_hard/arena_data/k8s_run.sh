#!/bin/bash

# ================= 配置区域 =================
INPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/arena_hard/data"
# 注意：你之前改了这个路径，确保它层级正确
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/arena_hard/data/responses"
MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-V3.2/DeepSeek-V3.2"
INFERENCE_SCRIPT="/mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/llm_judge/inference.py"

# Pod 列表
pods=(
    "dpo-data-3-gj6wd" "dpo-data-4-hl59h" "dpo-data-5-rp6nz" "dpo-data-6-blg5d"
    "dpo-data-7-ch5qw" "dpo-data-8-8cgfz" "dpo-data-9-9f6vs" "dpo-data-10-q9vm6"
    "dpo-data-11-64mbg" "dpo-data-12-bprcz" "dpo-data-13-92rkn" "dpo-data-14-zm5dl"
    "dpo-data-15-j4hzj" "dpo-data-16-4fhn9" "dpo-data-17-gwkkg" "dpo-data-18-25k2x"
    "dpo-data-19-q92sh" "dpo-data-20-2rtp6" "dpo-data-21-6pmsb" "dpo-data-22-t45rt"
)

echo "========================================================"
echo "阶段 1: 全局清理 (Killing sglang on all pods)"
echo "========================================================"

for pod in "${pods[@]}"; do
    echo "Cleaning $pod ..."
    # 只执行 kill，不执行 python。即使这里报 137 也没关系，因为我们的目的就是杀进程。
    kt exec "$pod" -- bash -c "pkill -9 -f sglang" || true
done

echo ""
echo "⏳ 等待 20 秒，让 Kubernetes 回收内存资源..."
sleep 20
echo "✅ 内存回收应该完成了。"
echo ""

echo "========================================================"
echo "阶段 2: 启动推理任务"
echo "========================================================"

for i in {0..19}; do
    pod_name=${pods[$i]}
    if [ -z "$pod_name" ]; then break; fi
    
    # 格式化分片 ID
    shard_suffix=$(printf "%02d" $i)
    input_file="${INPUT_DIR}/preprocess_10_times_part_${shard_suffix}.jsonl"
    output_file="${OUTPUT_DIR}/response_part_${shard_suffix}.jsonl"
    log_file="${OUTPUT_DIR}/log_part_${shard_suffix}.log"
    
    echo "启动 Pod: [$pod_name] -> Shard: $shard_suffix"

    # 核心命令：
    # 1. 先 mkdir 确保目录存在 (避免 No such file 报错)
    # 2. 这里的 sleep 1 是为了防止 nohup 在断开瞬间被杀
    cmd="mkdir -p ${OUTPUT_DIR}; \
         nohup python $INFERENCE_SCRIPT \
            --input \"$input_file\" \
            --output \"$output_file\" \
            --model_path $MODEL_PATH \
            --tp_size 1 \
            --dp_size 8 \
            --max_tokens 32768 \
         > \"$log_file\" 2>&1 & sleep 2"

    # 提交任务
    kt exec "$pod_name" -- bash -c "$cmd"

    # 简单检查
    echo "  -> 任务已提交"
    sleep 1
done

echo "========================================================"
echo "所有任务已分发。请用以下命令检查某个 Pod 的日志："
echo "kt exec ${pods[0]} -- tail -f ${OUTPUT_DIR}/log_part_00.log"