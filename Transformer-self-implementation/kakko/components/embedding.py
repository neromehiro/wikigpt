# python /app/Transformer-self-implementation/components/embedding.py

# フルーツのリスト
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry', 
          'fig', 'grape', 'honeydew', 'italian_lemon', 'jackfruit']

# フルーツと整数値をマッピングする辞書
fruit_to_int = {fruit: i for i, fruit in enumerate(fruits)}
print(fruit_to_int)

from tensorflow.keras import layers, models

def define_model(input_dim, embedding_dim):
    inputs = layers.Input(shape=(1,))
    x = layers.Embedding(input_dim=input_dim, output_dim=embedding_dim)(inputs)
    model = models.Model(inputs, x)  # このモデルでは特定のタスクは行わず、単にembedding結果を返します。
    return model

# モデルの定義。フルーツの数がinput_dim、各フルーツを表現するベクトルの次元数がembedding_dim
model = define_model(input_dim=len(fruits), embedding_dim=5)

import numpy as np

# 各フルーツのembedding結果を表示
for fruit, i in fruit_to_int.items():
    embedded_fruit = model.predict(np.array([[i]]))
    print(f"{fruit}: {embedded_fruit}")
