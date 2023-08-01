# ファイル名：/app/Transformer-self-implementation/kakko/components/test_transformer.py
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tensorflow.keras.models import load_model
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR メッセージのみを表示
import tensorflow as tf


# 元のトークンとIDの対応付け
tokens = ["(", ")", "[", "]", "{", "}"]
token2id = {token: i for i, token in enumerate(tokens)}
id2token = {i: token for i, token in enumerate(tokens)}

# モデルの読み込み先
model_path = "/app/Transformer-self-implementation/kakko/models/mymodel0.h5"

# モデルの読み込み
model = load_model(model_path)


# 括弧の組み合わせが正しいかどうかを確認する関数
def is_valid_bracket_sequence(seq):
    bracket_pairs = {'(': ')', '[': ']', '{': '}'}
    stack = []
    for bracket in seq:
        if bracket in bracket_pairs:  # 開始括弧
            stack.append(bracket)
        elif not stack or bracket_pairs[stack.pop()] != bracket:  # 閉じ括弧
            return False
    return not stack  

def predict_next_token(model, token_sequence, k):
    # 入力のトークンをIDに変換
    input_sequence = [token2id[token] for token in token_sequence]
    # モデルへの入力は (1, sequence_length) の形状である必要がある
    input_sequence = np.array(input_sequence)[np.newaxis, :]
    # モデルを使って予測を行う（verbose=0に設定して進捗ログを非表示にする）
    predictions = model.predict(input_sequence, verbose=0)
    # 予測されるIDがtokensリストの範囲内に収まるように修正
    predictions = predictions[0, :len(tokens)]
    # 最も確率の高いトークンのIDを取得
    top_k_ids = np.argsort(-predictions)[:k]  # Top-kトークンIDを取得
    predicted_token_id = np.random.choice(top_k_ids)  # Top-kの中からランダムに選択
    # IDをトークンに戻す
    predicted_token = id2token[predicted_token_id]
    return predicted_token



def generate_bracket_string(max_depth, length, bracket_types):
    assert max_depth <= len(bracket_types), "Not enough bracket types for the given depth"

    brackets = [bracket for bracket in bracket_types]
    stack = []
    string = ''
    
    for _ in range(length):
        # Reserve 1 space for closing bracket if necessary
        if len(string) + len(stack) >= length:
            break

        if not stack or (len(stack) < max_depth and np.random.choice([True, False])):
            depth = len(stack)  # Determine the bracket type based on depth
            stack.append(brackets[depth][0])  # Opening bracket
            string += brackets[depth][0]
        elif stack:  # Ensure stack is not empty before trying to pop
            # Find the depth of the last opened bracket
            depth = next(i for i, b in enumerate(brackets) if b[0] == stack[-1]) 
            stack.pop()  # Pop opening bracket
            string += brackets[depth][1]  # Closing bracket
    
    # Close all remaining open brackets
    while stack and len(string) < length:
        # Find the depth of the last opened bracket
        depth = next(i for i, b in enumerate(brackets) if b[0] == stack[-1]) 
        stack.pop()
        string += brackets[depth][1]
    
    return string

def test_model(model, num_samples, k):
    bracket_types = [('(', ')'), ('[', ']'), ('{', '}')]
    correct_predictions = 0
    total_predictions = 0
    
    # 出力ディレクトリの作成
    output_dir = "/app/Transformer-self-implementation/kakko/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 結果を保存するファイルパスを生成
    existing_files = os.listdir(output_dir)
    existing_output_files = [f for f in existing_files if f.startswith("output") and f.endswith(".txt")]
    new_file_id = len(existing_output_files) + 1
    file_path = f"{output_dir}/output{new_file_id}.txt"
    
    for sample_id in range(num_samples):
        string = generate_bracket_string(3, 30, bracket_types)
        prediction_output = ""
        correct_output = ""
        for i in range(len(string) - 1):
            token_sequence = list(string[:i+1])
            correct_next_token = string[i+1]
            predicted_token = predict_next_token(model, token_sequence, k)  # トップ-k サンプリングを使用
            prediction_output += predicted_token
            correct_output += correct_next_token
            # 予測が正しい括弧の組み合わせならば、正解としてカウント
            if is_valid_bracket_sequence(prediction_output):
                correct_predictions += 1
            total_predictions += 1

        # 各テストの正答率を計算
        accuracy = correct_predictions / total_predictions

        # 結果をファイルに保存
        try:
            with open(file_path, "a") as f:  # "a" モードで追記
                f.write(f"Test{sample_id + 1} Accuracy: {accuracy * 100}%\n")
                f.write(f"Input  : {string}\n")
                f.write(f"Output : {prediction_output}\n")
                f.write(f"Correct: {correct_output}\n\n")  # 追加: 各テストの結果を区切るための空行
        except Exception as e:
            print(f"Failed to write to file {file_path}. Error: {e}")

    # 全体の正答率を表示
    overall_accuracy = correct_predictions / total_predictions
    print(f"Model accuracy: {overall_accuracy * 100}%")

# test_model関数を呼び出す
test_model(model, 10, 5)
