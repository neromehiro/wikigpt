# file: generate/generate_text.py
from keras.models import load_model
import numpy as np
import pickle

def generate_text(model, id_to_token, seed_tokens, num_tokens_to_generate):
    # 生成するトークンのリストを初期化
    generated_tokens = list(seed_tokens)

    # 指定された数のトークンを生成
    for _ in range(num_tokens_to_generate):
        # 直近のトークンを入力シーケンスとして使用
        input_sequence = np.array(generated_tokens[-100:]).reshape(1, 100, 1)
        # 次のトークンを予測
        predicted_token_id = np.argmax(model.predict(input_sequence, verbose=0), axis=-1)[0]
        # 予測されたトークンをリストに追加
        generated_tokens.append(predicted_token_id)

    # トークンIDを実際のトークンに変換
    generated_text = ' '.join(id_to_token[token_id] for token_id in generated_tokens)

    return generated_text

# 使用例
model = load_model('my_model.h5')
with open('id_to_token.pkl', 'rb') as f:
    id_to_token = pickle.load(f)
with open('encoded_extracted/AA/wiki_00', 'r') as f:
    encoded_tokens = [int(line.strip()) for line in f]
seed_tokens = encoded_tokens[:100]
generated_text = generate_text(model, id_to_token, seed_tokens, num_tokens_to_generate=1000)
print(generated_text)
