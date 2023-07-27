# file: tokenize_text.py
from transformers import AutoTokenizer

tokenizer_jp = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

def tokenize_text(text, chunk_size=512):
    # テキストを一定の長さのチャンクに分割
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    decoded_tokens = []
    for chunk in text_chunks:
        # 各チャンクをトークン化
        tokenized_inputs = tokenizer_jp(chunk, padding=True, truncation=True, return_tensors='pt')
        # 入力IDを取り出し、それらをトークンにデコード
        decoded_chunk_tokens = [tokenizer_jp.decode(ids) for ids in tokenized_inputs['input_ids']]
        decoded_tokens.extend(decoded_chunk_tokens)

    return decoded_tokens
