# file: preprocess/run_encode_tokens.py
from encode_tokens import encode_tokens
import pickle
import os

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

# # 使用例
# run_encode_tokens("tokenize_extracted/AA/wiki_00", "encoded_extracted/AA/wiki_00")

folders = [chr(i) for i in range(ord('A'), ord('Q')+1)]  # AA to AQ
for folder1 in folders:
    for folder2 in folders:
        folder = folder1 + folder2  # AA to AQ
        num_files = 100 if folder != 'AQ' else 49  # AQ folder has 49 files
        for i in range(num_files):
            input_file = f"tokenize_extracted/{folder}/wiki_{str(i).zfill(2)}"
            output_file = f"encoded_extracted/{folder}/wiki_{str(i).zfill(2)}"
            run_encode_tokens(input_file, output_file)