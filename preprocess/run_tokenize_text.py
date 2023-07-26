# file: /preprocess/run_tokenize_text.py
# step2

from tokenize_text import tokenize_text
import os

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

# ディレクトリ名のリストを作成
dir_names = [chr(i) for i in range(ord('A'), ord('Q')+1)] # ['A', 'B', 'C', ..., 'Q']
file_names = [f"wiki_{i:02}" for i in range(100)]  # ['wiki_00', 'wiki_01', ..., 'wiki_99']

# 全てのディレクトリとファイルに対して処理を行う
for dir1 in dir_names:
    for dir2 in dir_names:
        folder_path = f"{dir1}{dir2}"
        input_folder = f"processed_extracted/{folder_path}"
        output_folder = f"tokenize_extracted/{folder_path}"
        os.makedirs(output_folder, exist_ok=True)
        
        for file_name in file_names:
            input_file_path = f"{input_folder}/{file_name}"
            output_file_path = f"{output_folder}/{file_name}"
            
            # ファイルが存在しなければ次のファイルへ
            if not os.path.isfile(input_file_path):
                continue
            
            run_tokenize_text(input_file_path, output_file_path)
            print(f"Completed: {output_file_path}")  # ログ出力
            
        # AQ/wiki_48 で終了するようにする
        if folder_path == "AQ" and file_name == "wiki_48":
            break

# # 使用例
# run_tokenize_text("processed_extracted/AA/wiki_00", "tokenize_extracted/AA/wiki_00")
