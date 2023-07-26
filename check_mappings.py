# file: check_mappings.py
import pickle

# token_to_idとid_to_tokenのマッピングをロード
with open('token_to_id.pkl', 'rb') as file:
    token_to_id = pickle.load(file)
with open('id_to_token.pkl', 'rb') as file:
    id_to_token = pickle.load(file)

# token_to_idの値が全て整数値の文字列であることを確認
for token, id in token_to_id.items():
    if not id.isdigit():
        print(f"Non-integer ID found for token {token}: {id}")
        break
else:
    # token_to_idの値を整数に変換
    token_to_id = {token: int(id) for token, id in token_to_id.items()}

    # マッピングが一貫性があることを確認
    for token, id in token_to_id.items():
        if id_to_token[id] != token:
            print(f"Inconsistent mapping: {token} maps to {id} but {id} maps to {id_to_token[id]}")
            break
    else:
        print("All mappings are consistent.")

    # 修正したマッピングを保存
    with open('token_to_id.pkl', 'wb') as file:
        pickle.dump(token_to_id, file)
