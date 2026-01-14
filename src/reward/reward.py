import re
import json
from typing import Dict, List
from pathlib import Path

from tqdm import tqdm
from src.reward.math_verify_reward import grade_answer
from src.utils import _calculate_matrics, _extract_answer

def _math_eval(
    item: Dict
) -> Dict:

    label = item.get("label", "")
    raw_eval_res = item.get("response", "") 

    pred_ans = _extract_answer(raw_eval_res)
    if pred_ans is None:
        item["pred"] = pred_ans
        item["score"] = (0.0, 0.0)
        return None, item

    score = grade_answer(f"${pred_ans}$", f"${label}$")

    item["pred"] = pred_ans
    item["score"] = score

    return item, None

def _ifeval_eval(
    item: Dict
) -> Dict:
    response = item.get("response", "")
    # TODO: Implement ifeval logic
    return None, item

def eval_results(
    eval_output_file: Path,
    final_eval_output_file: Path
) -> Dict[str, Dict[str, float]]:
    
    updated_lines = []
    updated_items = []
    failed_items = []

    # Get total lines for progress bar
    with open(eval_output_file, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    with open(eval_output_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Evaluating results"):
            if not line.strip(): 
                continue
            
            item = json.loads(line)
            eval_type = item.get("eval_type", "math")
           
            updated_item, failed_item = None, None
            if eval_type == "math":
                updated_item, failed_item = _math_eval(item)
            elif eval_type == "ifeval":
                updated_item, failed_item = _ifeval_eval(item)
            
            if failed_item is not None:
                failed_items.append(failed_item)
                updated_items.append(failed_item)
                updated_lines.append(json.dumps(failed_item, ensure_ascii=False))
            elif updated_item is not None:
                updated_items.append(updated_item)
                updated_lines.append(json.dumps(updated_item, ensure_ascii=False))

    # ------------------ write the updated items to the file ------------------ 
    new_eval_output_file = eval_output_file.with_suffix(f".scored{eval_output_file.suffix}")
    with open(new_eval_output_file, "w", encoding="utf-8") as f:
        for line in updated_lines:
            item = json.loads(line)
            cleaned_item = {k: v for k, v in item.items() if k not in ["prompt", "response"]}
            f.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
    
    # ------------------ write the failed items to the file ------------------ 
    with open(eval_output_file.with_suffix(f".failed{eval_output_file.suffix}"), "w", encoding="utf-8") as f:
        for item in failed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ------------------ calculate the metrics and return ------------------ 
    results = _calculate_matrics(updated_items)
    with open(final_eval_output_file, "w", encoding="utf-8") as f:
        for item in results.items():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return results

if __name__ == "__main__":
    test_res = "The answer is \\boxed{m/frac{2}{3}}"
    print(f"Extracted: {extract_answer(test_res)}")
    print(f"Reward: {get_reward(extract_answer(test_res), 'm/frac{{2}}{{3}}', 'dapo')}")
