# tensorflowとkerasをインポート
#  python /app/check/model_parameters.py
import tensorflow as tf
from tensorflow import keras

# .h5形式のモデルを読み込む
model = keras.models.load_model('/app/models/mymodel4.h5')

# モデルの概要を表示する
model.summary()
