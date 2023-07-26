# file: run_tokenize_text.py
from tokenize_text import tokenize_text

def run_tokenize_text(input_file_path, output_file_path):
    # ファイルからテキストを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # テキストをトークン化
    tokens = tokenize_text(text)

    # トークンをファイルに保存
    with open(output_file_path, "w", encoding="utf-8") as file:
        for token in tokens:
            file.write(token + '\n')

# 使用例
run_tokenize_text("processed_extracted/AA/wiki_00", "tokenize_extracted/AA/wiki_00")
