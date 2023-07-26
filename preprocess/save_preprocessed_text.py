# file: save_preprocessed_text.py
from preprocess_text import preprocess_text
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

# 使用例
save_preprocessed_text("extracted/AA/wiki_00", "processed_extracted/AA/wiki_00")
