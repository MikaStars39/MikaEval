OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/arena_hard/data"
MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-V3.2/DeepSeek-V3.2"

SCRIPT_DIR="/mnt/llm-train/users/explore-train/qingyu/slimulation"

function preprocess() {
    python $SCRIPT_DIR/recipe/arena_hard/arena_data/preprocess.py \
        --input $OUTPUT_DIR/question_en.jsonl \
        --output $OUTPUT_DIR/preprocess.jsonl \
        --tokenizer $MODEL_PATH
}

function inference() {
    python $SCRIPT_DIR/recipe/arena_hard/inference.py \
        --input "$OUTPUT_DIR/preprocess.jsonl" \
        --output "$OUTPUT_DIR/results.jsonl" \
        --model "$MODEL_PATH" \
        --tp-size 8 \
        --dp-size 1 \
        --temperature 1 \
        --top-p 1 \
        --max-tokens 32768
}

function extract_questions() {
    python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/arena_data/extract_questions.py \
        --input $OUTPUT_DIR/results.jsonl \
        --output $OUTPUT_DIR/question.jsonl \
        --max-per-seed 10
}

function preprocess_again() {
    python $SCRIPT_DIR/recipe/arena_hard/arena_data/preprocess.py \
        --input $OUTPUT_DIR/question.jsonl \
        --output $OUTPUT_DIR/preprocess_10_times.jsonl \
        --tokenizer $MODEL_PATH
}

function inference() {
    python $SCRIPT_DIR/recipe/arena_hard/inference.py \
        --input "$OUTPUT_DIR/preprocess_10_times.jsonl" \
        --output "$OUTPUT_DIR/results_10_times.jsonl" \
        --model "$MODEL_PATH" \
        --tp-size 8 \
        --dp-size 1 \
        --temperature 1 \
        --top-p 1 \
        --max-tokens 32768
}

MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507"
function inference_qwen() {
    python $SCRIPT_DIR/recipe/arena_hard/arena_data/preprocess_qwen.py \
        --input $OUTPUT_DIR/question_en.jsonl \
        --output $OUTPUT_DIR/preprocess_qwen.jsonl \
        --tokenizer $MODEL_PATH

    python $SCRIPT_DIR/recipe/arena_hard/inference.py \
        --input "$OUTPUT_DIR/preprocess_qwen.jsonl" \
        --output "$OUTPUT_DIR/results_qwen.jsonl" \
        --model "$MODEL_PATH" \
        --tp-size 1 \
        --dp-size 8 \
        --temperature 1 \
        --top-p 1 \
        --max-tokens 32768
}