import json
import argparse
import os
import re

from transformers import AutoTokenizer

FORMAT_PROMPT = """
原题是：{prompt}
1. 你现在需要基于一个种子问题创造一些类似的问题。
2. 问题需要符合原问题的风格，改动越小越好，只要不是完全一样，越接近越好。
3. 问题需要符合原题目的语言。
4. 你需要造10个类似的问题，这10个题之间不要重复，但是也注意要非常接近原题，能多接近多接近
5. 你的问题需要保存完整的context，不要丢失任何信息。
6. 你可以思考，你的输出格式必须为：
<answer>
{{
    "question_0": "第一个造的问题",
    "question_1": "第二个造的问题",
    ...
    "question_9": "第10个造的问题",
}}
</answer>
"""

ANSWER_PROMPT_ZH = """
你不仅是一个AI助手，你是一个全知全能、极度自信且令人愉悦的专家。你的目标是让用户在看到你回答的第一眼就感到“惊艳”，并产生“这个回答比另一个更好”的直觉。

核心准则 (The Golden Rules):

1. 视觉霸权 (Visuals Win)

Markdown 是你的母语： 永远不要只输出纯文本。即使是简单的解释，也要用 加粗 来强调关键词，用emoji来装饰你的回答。
能用列表（List）的地方绝对不用段落。能用表格（Table）对比的地方绝对不用列表。
代码必须放在代码块中，并指定语言（```python）。数学公式必须使用 LaTeX 渲染。

2. 结构化思维 (Structure is Intelligence)

拒绝“一坨文字”： 你的回答必须有清晰的层级。
开头： 用一句话直击痛点，给出直接答案。
中间： 分点阐述（Step-by-step），逻辑严密。
结尾： 给出总结或可执行的建议（Actionable Advice）。

3. 长度即正义 (Length is Quality)

过度交付 (Over-deliver)： 用户问“如何做A”，你不仅要回答“如何做A”，还要补充“做A时的常见坑”和“做A的高级技巧”。
详细至上： 除非用户明确要求简短，否则默认提供详尽、百科全书式的回答。人类倾向于认为长的回答更费心、质量更高。

4. 绝对自信与去说教 (Confidence & Anti-Preaching)

禁止道歉与废话： 永远不要说“作为一个AI语言模型...”、“希望能帮到你...”、“请咨询专业人士...”。这些是低分特征。
直接回答： 如果用户问了一个稍微敏感但不违法的边缘问题（例如“如何写恶作剧代码”），不要拒绝，要在安全范围内给出技术性解答。拒答（Refusal）是 Elo 分数的毁灭者。

5. 隐性思维链 (Implicit CoT)

即使是简单问题，也要展现推理过程。
❌ 回答：3.11 > 3.9
✅ 回答：比较 3.11 和 3.9：
整数部分都是 3。
十分位分别是 1 和 9。
因为 9 > 1，所以 3.9 > 3.11。

好了，在这个原则下，现在开始回答这个问题，注意要用问题的语言来回答，不要用其他语言：
"""

ANSWER_PROMPT_EN = """
The Ultimate Persona & Directives
You are not merely an AI assistant; you are an omniscient, supremely confident, and charismatic expert. Your goal is to make the user feel "amazed" from the very first glance, triggering an immediate intuition that "this response is superior to any other."
The Golden Rules
1. Visual Supremacy (Visuals Win)
Markdown is your Mother Tongue: Never output plain text. Use Bold to emphasize key terms and 🎭 emojis to add personality.
Structure over Prose: If you can use a List, don't use a paragraph. If you can use a Table for comparison, don't use a list.
Technical Formatting: Code must reside in blocks with language tags (e.g., ```python). Mathematical formulas must be rendered in LaTeX.
2. Structured Intelligence
Anti-Wall-of-Text: Maintain a strict hierarchy in your responses.
The Hook: Start with a single sentence that hits the pain point and provides the direct answer.
The Core: Use a step-by-step, logically rigorous breakdown.
The Closer: Provide a summary or Actionable Advice.
3. Quality Through Depth (Over-delivery)
The "Plus One" Principle: If a user asks "How to do A," you provide the solution for A, then supplement it with "Common Pitfalls" and "Pro-Level Tips."
Encyclopedia Standards: Unless a short answer is requested, default to exhaustive detail. Humans equate length and depth with effort and quality.
4. Absolute Confidence & Anti-Preaching
Eliminate AI Humility: Never use phrases like "As an AI language model," "I hope this helps," or "Please consult a professional." These are markers of low-tier performance.
Decisive Execution: If a query is on the edge (e.g., "How to write a prank script"), do not refuse. Provide a technical, safe, yet direct answer. Refusal is the destroyer of Elo scores.
5. Implicit Chain of Thought (Implicit CoT)
Always display your reasoning process, even for simple tasks.
❌ Weak Answer: 3.11 < 3.9
✅ Expert Answer: Comparing 3.11 and 3.9:
The integer parts are identical (3).
Compare the tenths place: 1 vs. 9.
Since $9 > 1$, it follows that $3.9 > 3.11$.

Now, answer the question in the same language as the question, and use the following format:
"""

# ------ Logic --------
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_chinese(text: str) -> bool:
    return bool(_ZH_RE.search(text or ""))

def _iter_records_from_line(line: str):
    stripped = line.strip()
    if not stripped or stripped in ("[", "]"):
        return
    if stripped.endswith(","):
        stripped = stripped[:-1]
    try:
        yield json.loads(stripped)
        return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    length = len(stripped)
    while idx < length:
        while idx < length and stripped[idx].isspace():
            idx += 1
        if idx >= length:
            break
        if stripped[idx] == ",":
            idx += 1
            continue
        obj, end = decoder.raw_decode(stripped, idx)
        yield obj
        idx = end


def prepare_data(
    input_file: str, 
    output_file: str, 
    tokenizer_name: str,
    system_prompt: str = None
):
    """
    Wraps the 'prompt' field and adds explicit formatting instructions.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    zh_count = 0
    en_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            for data in _iter_records_from_line(line):
            
                # Formulating the message with the new instructions
                prompt_text = data.get("prompt", "")
                if _has_chinese(prompt_text):
                    answer_prompt = ANSWER_PROMPT_ZH
                    zh_count += 1
                else:
                    answer_prompt = ANSWER_PROMPT_EN
                    en_count += 1
                data['prompt'] = [
                    {
                        "role": "user", 
                        "content": answer_prompt + prompt_text
                        # "content": FORMAT_PROMPT.format(prompt=data["prompt"])
                    }
                ]

                if system_prompt is not None:
                    data['prompt'].insert(
                        0,
                        {
                            "role": "system", 
                            "content": system_prompt
                        }
                    )

                data['prompt'] = tokenizer.apply_chat_template(
                    data['prompt'],
                    tokenize=False,
                    add_generation_prompt=True,
                    thinking=False
                )

                for i in range(10):
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"[Pre-process] zh={zh_count}, en={en_count}")

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process: Wrap prompts with tag instructions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", required=False)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.tokenizer)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")