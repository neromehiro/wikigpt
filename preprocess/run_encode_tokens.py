# file: preprocess/run_encode_tokens.py
# python /app/preprocess/run_encode_tokens.py
from encode_tokens import encode_tokens
import pickle
import os
from config import BASE_DIR, TOKENIZE_DIR, ENCODE_DIR

def run_encode_tokens(input_file_path, output_file_path):
    # ファイルからトークンを読み込む
    with open(input_file_path, "r", encoding="utf-8") as file:
        tokens = [line.strip() for line in file]

    # トークンをエンコード
    encoded_tokens, id_to_token, token_to_id = encode_tokens(tokens)
    
    # 出力ディレクトリが存在することを確認し、存在しない場合は作成
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # エンコードしたトークンをファイルに保存
    with open(output_file_path, "w", encoding="utf-8") as file:
        for encoded_token in encoded_tokens:
            file.write(str(encoded_token) + '\n')

    # トークンとIDのマッピングを保存
    with open("token_to_id.pkl", "wb") as file:
        pickle.dump(token_to_id, file)
    
    # IDとトークンのマッピングを保存
    with open("id_to_token.pkl", "wb") as file:
        pickle.dump(id_to_token, file)
        
    print(f"Completed: {output_file_path}")

# トークン化済みのファイルを取得
input_base_dir = TOKENIZE_DIR
output_base_dir = ENCODE_DIR

# 全てのファイルに対してエンコードを行う
for root, dirs, files in os.walk(input_base_dir):
    for file_name in files:
        input_file_path = os.path.join(root, file_name)
        relative_path = os.path.relpath(input_file_path, input_base_dir)
        output_file_path = os.path.join(output_base_dir, relative_path)

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        run_encode_tokens(input_file_path, output_file_path)