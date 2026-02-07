import sys
import json
import argparse
import re
from typing import Dict, Any

# ------ Detection Logic --------

def is_chinese(text: str) -> bool:
    """Check if the text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def is_english(text: str) -> bool:
    """Check if the text is primarily English/Latin characters."""
    clean_text = re.sub(r'[\s\d\W_]+', '', text)
    if not clean_text:
        return False
    
    latin_chars = re.findall(r'[a-zA-Z]', clean_text)
    # Heuristic: 80% or more Latin characters
    return len(latin_chars) / len(clean_text) > 0.8

def is_valid_lang(text: str) -> bool:
    """Validate if the string is either Chinese or English."""
    target = text.strip()
    if not target:
        return False
    return is_chinese(target) or is_english(target)

# ------ Filtering Engine --------

def filter_jsonl(input_stream, output_stream, key: str):
    """
    Reads from input, validates the specified key, 
    and writes valid JSON lines to output.
    """
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            content = str(data.get(key, ""))
            
            # Only pipe through the lines that match our language criteria
            if is_valid_lang(content):
                # Ensure the output is valid JSON text
                output_stream.write(json.dumps(data, ensure_ascii=False) + '\n')
                
        except json.JSONDecodeError:
            # Errors are sent to stderr to keep stdout clean
            print(f"Skipping invalid JSON line", file=sys.stderr)

# ------ CLI Configuration --------

def main():
    parser = argparse.ArgumentParser(
        description="Filter a JSONL file to keep only English and Chinese entries."
    )
    
    # Positionals/Optional arguments for files
    parser.add_argument(
        "input", 
        nargs="?", 
        type=argparse.FileType("r"), 
        default=sys.stdin,
        help="Input JSONL file path (default: stdin)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output JSONL file path (default: stdout)"
    )
    
    parser.add_argument(
        "--key", "-k",
        type=str, 
        default="prompt", 
        help="The JSON key to check language for (default: 'prompt')"
    )

    args = parser.parse_args()

    # ------ Execution --------
    try:
        filter_jsonl(args.input, args.output, args.key)
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        # Close files properly
        if args.input is not sys.stdin:
            args.input.close()
        if args.output is not sys.stdout:
            args.output.close()

if __name__ == "__main__":
    main()