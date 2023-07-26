# file: encode_tokens.py
from nltk import FreqDist
from tokenize_text import tokenize_text

def encode_tokens(tokens):
    # トークンの頻度分布を計算
    freq_dist = FreqDist(tokens)

    # 各トークンを一意の整数にエンコード
    token_to_id = {token: i for i, token in enumerate(freq_dist.keys())}
    encoded_tokens = [token_to_id[token] for token in tokens]

    return encoded_tokens

# 使用例
# text = "ここに前処理済みのテキストを入力"
# tokens = tokenize_text(text)
# encoded_tokens = encode_tokens(tokens)
# print(encoded_tokens)
