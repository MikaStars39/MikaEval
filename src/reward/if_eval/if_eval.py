from niu_vllm import LLM
from config import ifeval_path, model_path
import datasets
import json
import os
from utils.instructions_registry import INSTRUCTION_DICT

def make_prompt(instance):
    prompt = instance['prompt']
    return {'prompt': [{'role': 'user', 'content': prompt}]}

def judge(instance):
    instructions = instance['instruction_id_list']
    kwargs_list = instance['kwargs']
    response = instance['response']

    prompt_level_pass_flag = True
    instruction_pass_cnt = 0
    
    for instruction_id, kwargs in zip(instructions, kwargs_list):
        instruct = INSTRUCTION_DICT[instruction_id](instruction_id)
        
        # 获取该指令支持的参数键
        supported_keys = instruct.get_instruction_args_keys()
        
        # 只传递支持的参数
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}
        
        instruct.build_description(**filtered_kwargs)
        passed = instruct.check_following(response)
        
        if passed:
            instruction_pass_cnt += 1
        else:
            prompt_level_pass_flag = False
    
    return {
        'instruction_count': len(instructions),
        'instruction_pass_cnt': instruction_pass_cnt,
        'all_passed': prompt_level_pass_flag
    }


def calculate_scores(results):
    """计算prompt level和instruct level的分数"""
    total_prompts = len(results)
    prompt_level_passed = sum(1 for r in results if r['all_passed'])
    
    total_instructions = sum(r['instruction_count'] for r in results)
    instruction_level_passed = sum(r['instruction_pass_cnt'] for r in results)
    
    prompt_level_score = prompt_level_passed / total_prompts if total_prompts > 0 else 0
    instruct_level_score = instruction_level_passed / total_instructions if total_instructions > 0 else 0
    
    return {
        'prompt_level_score': prompt_level_score,
        'instruct_level_score': instruct_level_score,
        'prompt_level_passed': prompt_level_passed,
        'total_prompts': total_prompts,
        'instruction_level_passed': instruction_level_passed,
        'total_instructions': total_instructions
    }

if __name__ == '__main__':
    
    ds = datasets.load_dataset("/mnt/llm-train/users/explore-train/qingyu/.cache/IFEval", split="train")
    ds = ds.map(make_prompt)

    # 检查临时结果文件是否存在
    temp_result_path = './temp_result.json'
    if os.path.exists(temp_result_path):
        print("发现临时结果文件，直接加载...")
        with open(temp_result_path, 'r', encoding='utf-8') as f:
            outputs = json.load(f)
    else:
        print("临时结果文件不存在，开始调用vllm进行推理...")
        llm = LLM(model_path)
        outputs = llm.inference(ds['prompt'], max_tokens=8192)
        
        # 保存临时结果
        with open(temp_result_path, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
        print("推理完成，结果已保存到临时文件")
    
    # 添加response列并进行评判
    ds = ds.add_column('response', outputs)
    ds = ds.map(judge, num_proc=16)
    
    # 计算分数
    results = list(ds)
    scores = calculate_scores(results)
    
    # 输出结果
    print("=== IFEval 评估结果 ===")
    print(f"Prompt Level Score: {scores['prompt_level_score']:.4f} ({scores['prompt_level_passed']}/{scores['total_prompts']})")
    print(f"Instruction Level Score: {scores['instruct_level_score']:.4f} ({scores['instruction_level_passed']}/{scores['total_instructions']})")
    
    # 保存详细结果
    final_results = {
        'scores': scores,
        'detailed_results': results
    }
    
    # with open('./ifeval_results.json', 'w', encoding='utf-8') as f:
    #     json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    # print("详细结果已保存到 ifeval_results.json")