# file: /preprocess/save_preprocessed_text.py
# step1
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

# テスト
# save_preprocessed_text("extracted/AA/wiki_00", "processed_extracted/AA/wiki_00")


# 本番
 
# すべてのフォルダとファイルに対して前処理を適用
folders = ["AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP", "AQ"]

for folder in folders:
    for i in range(100):
        # 'AQ' のときだけ 'wiki_48' まで
        if folder == "AQ" and i > 48:
            break

        input_file_path = f"extracted/{folder}/wiki_{str(i).zfill(2)}"
        output_file_path = f"processed_extracted/{folder}/wiki_{str(i).zfill(2)}"
        
        # 入力ファイルが存在する場合だけ処理を行う
        if os.path.exists(input_file_path):
            save_preprocessed_text(input_file_path, output_file_path)