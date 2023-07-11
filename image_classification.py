### 事前準備 ###
#
# ①venvなどを使って新しい仮想環境を作成する　※クリーンな環境にて制作したいため
#
# ②添付した「requirements.txt」を使って、必要なライブラリを一括インストールする
#   例）python -m pip install -r requirements.txt
#
# 参考：requirements.txt の内容
#   streamlit             ← 毎度、おなじみ
#   typing_extensions     ← これがないとエラーが出ることがあるので一応
#   numpy                 ← 毎度、おなじみ
#   pandas                ← 毎度、おなじみ
#   tensorflow-cpu        ← 深層学習用ライブラリ（容量の関係で今回はCPU版を採用）
#

# ライブラリのインポート
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


# 画像(img)が属するクラスを推論する関数（この関数は変更する必要なし）
def teachable_machine_classification(img, weights_file):

    # モデルの読み込み
    model = load_model(weights_file)

    # kerasモデルに投入するのに適した形状の配列を作成する。
    # 配列に入れることができる画像の「長さ」または枚数は
    # shapeタプルの最初の位置（この場合は1）で決まる。
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # これを画像へのパスに置き換える
    # image = Image.open(img)
    image = img

    # Teachable Machineと同じ方法で、224x224にリサイズする。
    # 少なくとも224x224になるように画像をリサイズし、中心から切り取る。
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # 画像をnumpyの配列に変換する
    image_array = np.asarray(image)

    # 画像の正規化
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 画像を配列に読み込む
    data[0] = normalized_image_array

    # 推論を実行する
    prediction = model.predict(data)

    # 推論結果をメインモジュールに戻す
    return prediction.tolist()[0]


# メインモジュール
def main():

    # タイトルの表示
    st.title("Image Classification with Google's Teachable Machine")

    # アップローダの作成
    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")

    # 画像がアップロードされた場合...
    if uploaded_file is not None:

        # 画像を画面に表示
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # teachable_machine_classification関数に画像を引き渡してクラスを推論する
        prediction = teachable_machine_classification(image, 'keras_model.h5')        
        st.caption(f'推論結果：{prediction}番') # 戻り値の確認（デバッグ用）

        classNo = np.argmax(prediction)          # 一番確率の高いクラス番号を算出
        st.caption(f'判定結果：{classNo}番')      # 戻り値の確認（デバッグ用）

        # 推論の確率を小数点以下3桁で丸め×100(%に変換)
        pred0 = round(prediction[0],3) * 100  # 猫の確率(%)
        pred1 = round(prediction[1],3) * 100  # 犬の確率(%)
        pred2 = round(prediction[2],3) * 100  # アザラシの確率(%)

        # 推論で得られたクラス番号(初期値は0)によって出力結果を分岐
        if classNo == 0:
            st.subheader(f"これは{pred0}％の確率で「猫」です！")
        elif classNo == 1:
            st.subheader(f"これは{pred1}％の確率で「犬」です！")
        elif classNo == 2:
            st.subheader(f"これは{pred2}％の確率で「アザラシ」です！")


# mainの起動
if __name__ == "__main__":
    main()


#=============================================================================#
#
# 課題. 画像分類ツールの作成
#
# 初級（必須）：ファイルの冒頭にある「事前準備」を行って、プログラムを稼働させよう
# ※無事、正しく稼働すると「犬」または「猫」の画像を認識するプログラムが起動します
#
# 中級（任意）：犬、猫に加えて「アザラシ」を認識できるよう新しいモデルを作成しよう
# ※下段欄外の「Teachable Machineでモデルを作成する方法（中級以上）」を参照のこと
#
# 上級（任意）：画像認識を利用した、オリジナルのWebアプリを作りましょう（自由課題）
#              アイデア1) マニアックな代物を認識してくれる「◯◯鑑定士」アプリ
#              アイデア2) カメラで撮影した顔が、何の動物に似ているか判定するアプリ
#              ※Streamlitの各種API、コンポーネントなどは、自由に使って結構です
# 
# #-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#
#
# 提出方法：中級以上は、モデルファイル（keras_model.h5）も一緒に提出してください
#           上級はアプリの概要や使い方などを簡単に下のコメント欄にご記入ください
#
# 氏名：石田宙
# 学科：高度情報工学
# 学年：4年
# コメント：◯◯◯◯（課題に取り組んだ感想や不明点、工夫した点など記入してください）
#
#=============================================================================#

### Teachable Machineでオリジナルの画像認識モデルを作成する方法（中級以上） ###
#
# 「Teachable Machine」の基本的な使い方は、以下のサイト等々を参考にしてください
# 参考リンク→　https://mekurun.com/tips/teachablemachine/
# 
# ※推奨ブラウザ：Google Chrome（他のブラウザだと、ファイルのD&Dができないかも）
# 
# 1. Teachable Machineにアクセス→「使ってみる」→「画像プロジェクト」→「標準の～」
#   https://teachablemachine.withgoogle.com/
#   
# 2. 各Classの「アップロード」ボタンからデータを追加する
# 
# 3. トレーニングを実行する
#
# 4. モデルをエクスポートする→「Tensorflow」タブ→「Keras」→「モデルをダウンロード」
#
# 5. モデルのファイル（converted_keras.zip）がダウンロードされる
# 
# 6. converted_keras.zip に含まれる「keras_model.h5」を、
#    この image_classification ファイルと「同じフォルダ」に格納する
#
# 7. 準備OK！
# 
# （以後、モデルを作り直すたびに、エクスポート → .h5ファイルの上書きをしてください）
