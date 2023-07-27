# file: /preprocess/run_tokenize_text.py
# python /app/preprocess/run_tokenize_text.py

from tokenize_text import tokenize_text
import os
from config import PREPROCESS_DIR, TOKENIZE_DIR

def run_tokenize_text(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    tokens = tokenize_text(text)

    with open(output_file_path, "w", encoding="utf-8") as file:
        for token in tokens:
            # スペースを改行に変更
            file.write(token.replace(' ', '\n') + '\n')

input_base_dir = PREPROCESS_DIR
output_base_dir = TOKENIZE_DIR

for root, dirs, files in os.walk(input_base_dir):
    for file_name in files:
        input_file_path = os.path.join(root, file_name)
        relative_path = os.path.relpath(input_file_path, input_base_dir)
        output_file_path = os.path.join(output_base_dir, relative_path)

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        run_tokenize_text(input_file_path, output_file_path)
