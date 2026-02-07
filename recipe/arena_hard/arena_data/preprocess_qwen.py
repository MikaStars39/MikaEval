import json
import argparse
import os
import re

from transformers import AutoTokenizer

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
                messages = [
                    {"role": "user", "content": prompt_text}
                ]
                data['prompt'] = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

# ------ CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process: Wrap prompts with tag instructions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.tokenizer, system_prompt=None)
    print(f"[Pre-process] Done. Inference file ready: {args.output}")