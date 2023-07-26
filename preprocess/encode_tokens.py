# file: encode_tokens.py
from tokenize_text import tokenize_text

def encode_tokens(tokens):
    # 一意のトークンを取得
    unique_tokens = list(set(tokens))

    # トークンをその一意なインデックスに基づいて数値にエンコード
    encoded_tokens = [unique_tokens.index(token) for token in tokens]

    return encoded_tokens

# 使用例
# text = "ここに前処理済みのテキストを入力"
# tokens = tokenize_text(text)
# encoded_tokens = encode_tokens(tokens)
# print(encoded_tokens)
