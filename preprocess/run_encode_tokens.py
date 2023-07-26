# file: preprocess/run_encode_tokens.py
from encode_tokens import encode_tokens
import pickle

def run_encode_tokens(input_file_path, output_file_path):
    # ファイルからトークンを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        tokens = [line.strip() for line in file]

    # トークンをエンコード
    encoded_tokens, token_to_id = encode_tokens(tokens)

    # id_to_token マッピングを作成
    id_to_token = {id: token for token, id in token_to_id.items()}

    # id_to_token マッピングを保存
    with open('id_to_token.pkl', 'wb') as f:
        pickle.dump(id_to_token, f)

    # エンコードしたトークンをファイルに保存
    with open(output_file_path, "w", encoding="utf-8") as file:
        for encoded_token in encoded_tokens:
            file.write(str(encoded_token) + '\n')

# 使用例
run_encode_tokens("tokenize_extracted/AA/wiki_00", "encoded_extracted/AA/wiki_00")
