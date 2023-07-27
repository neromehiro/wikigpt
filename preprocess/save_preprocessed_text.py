# file: /preprocess/save_preprocessed_text.py
# python /app/preprocess/save_preprocessed_text.py
# step1
from preprocess_text import preprocess_text
from config import BASE_DIR, PREPROCESS_DIR
import os

def save_preprocessed_text(input_file_path, output_file_path):
    # テキストデータの前処理
    processed_text = preprocess_text(input_file_path)

    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # 前処理したテキストデータをファイルに保存
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(processed_text)

# originalのディレクトリの中の全てのファイルを取得
input_base_dir = BASE_DIR
output_base_dir = PREPROCESS_DIR

# 全てのファイルに対して前処理を行う
for root, dirs, files in os.walk(input_base_dir):
    for file_name in files:
        input_file_path = os.path.join(root, file_name)
        relative_path = os.path.relpath(input_file_path, input_base_dir)
        output_file_path = os.path.join(output_base_dir, relative_path)

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        save_preprocessed_text(input_file_path, output_file_path)

# for folder in folders:
    # for i in range(100):
    #     # 'AQ' のときだけ 'wiki_48' まで
    #     if folder == "AQ" and i > 48:
    #         break

    #     input_file_path = f"extracted/{folder}/wiki_{str(i).zfill(2)}"
    #     output_file_path = f"processed_extracted/{folder}/wiki_{str(i).zfill(2)}"
        
    #     # 入力ファイルが存在する場合だけ処理を行う
    #     if os.path.exists(input_file_path):
    #         save_preprocessed_text(input_file_path, output_file_path)