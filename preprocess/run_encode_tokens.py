# file: run_encode_tokens.py
from encode_tokens import encode_tokens

def run_encode_tokens(input_file_path, output_file_path):
    # ファイルからトークンを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        tokens = [line.strip() for line in file]

    # トークンをエンコード
    encoded_tokens = encode_tokens(tokens)

    # エンコードしたトークンをファイルに保存
    with open(output_file_path, "w", encoding="utf-8") as file:
        for encoded_token in encoded_tokens:
            file.write(str(encoded_token) + '\n')

# 使用例
run_encode_tokens("tokenize_extracted/AA/wiki_00", "encoded_extracted/AA/wiki_00")
