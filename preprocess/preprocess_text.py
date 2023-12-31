# file: /preprocess/preprocess_text.py
from bs4 import BeautifulSoup
import re

def preprocess_text(file_path):
    # ファイルを読み込み
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # HTMLタグを取り除く
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text()

    # 特殊文字を取り除く
    text = re.sub(r'&lt;|&gt;|&amp;|&quot;', '', text)

    # 「」内の文字列を削除
    text = re.sub(r'「[^」]*」', '', text)

    # < ~~>内の文字列を削除
    text = re.sub(r'<[^>]*>', '', text)

    # "}}。" を削除
    text = text.replace('}}。', '')

    # "詳細はを参照" を削除
    text = text.replace('詳細はを参照', '')

    return text
