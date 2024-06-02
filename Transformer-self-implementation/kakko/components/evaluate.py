# python evaluate.py
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from typing import List

# ディレクトリ設定
dirs = {
    "original": "./dataset/original",
    "tokenize": "./dataset/tokenize",
    "preprocessed": "./dataset/preprocessed",
}

# モデルの保存パス
model_save_path = "/app/Transformer-self-implementation/kakko/models/best_model.h5"


# テストデータの保存パス
test_data_path = os.path.join(dirs["original"], "test_bracket_dataset.json")

# 評価結果の保存パス
evaluation_result_path = "evaluation_result.txt"

# トークンとIDを対応付ける辞書
tokens = ["(", ")", "【", "】", "{", "}", "input", ",output", ","]
token2id = {token: i for i, token in enumerate(tokens)}
id2token = {i: token for token, i in token2id.items()}

# モデルのロード
model = load_model(model_save_path)

def load_dataset(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def tokenize_string(string: str) -> List[str]:
    tokens = []
    current_token = ""
    for char in string:
        if char in token2id:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens

def preprocess_input(input_seq: str) -> List[int]:
    tokens = tokenize_string(input_seq)
    return [token2id[token] for token in tokens if token in token2id]

def decode_output(output_seq: List[int]) -> str:
    return "".join([id2token[id] for id in output_seq if id in id2token])

def evaluate_model(model, test_data: List[str]):
    correct_predictions = 0
    results = []

    for data in test_data:
        input_seq = data.split(",")[0].split(":")[1]
        expected_output = data.split(",")[1].split(":")[1]

        # 前処理
        preprocessed_input = preprocess_input(input_seq)
        preprocessed_input = np.array(preprocessed_input).reshape(1, -1)  # モデルの入力形状に合わせる

        # モデルの予測をシーケンス全体に渡って行う
        predicted_output_ids = []
        for _ in range(len(expected_output)):
            predicted_output = model.predict(preprocessed_input)
            predicted_id = np.argmax(predicted_output, axis=-1).flatten()[0]  # 予測結果をIDに変換
            predicted_output_ids.append(predicted_id)
            preprocessed_input = np.append(preprocessed_input, [[predicted_id]], axis=1)

        # デコード
        predicted_output = decode_output(predicted_output_ids)

        # 結果の保存
        results.append(f"Input: {input_seq}, Predicted Output: {predicted_output}, Expected Output: {expected_output}")
        
        if predicted_output == expected_output:
            correct_predictions += 1

    # 結果のファイル保存
    with open(evaluation_result_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result + "\n")

    accuracy = correct_predictions / len(test_data)
    return accuracy

# テストデータのロード
test_data = load_dataset(test_data_path)

# モデルの評価
accuracy = evaluate_model(model, test_data)
print(f"モデルの精度: {accuracy * 100:.2f}%")
print(f"評価結果は {evaluation_result_path} に保存されました。")
