# coding:utf-8 Copy Right Atelier Grenouille © 2015 -
#
# refer https://qiita.com/hiroeorz@github/items/2fbb3b8d12b0e20f0384#_reference-99be2aef3ed42fdd155a


from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# 学習用のデータを作る.
image_list = []
label_list = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("/home/pi/2018.03.19/"):
    if dir == ".DS_Store":
        continue

    dir1 = "/home/pi/2018.03.19/" + dir 
    label = 0

    if dir == "back":    # 稀勢の里はラベル0
        label = 0
    elif dir == "D-2":   # 橋本環奈はラベル1
        label = 1
    elif dir == "D-3":   # 橋本環奈はラベル1
        label = 2
    elif dir == "D-4":   # 橋本環奈はラベル1
        label = 3
    elif dir == "window":      # 広瀬すずはラベル2
        label = 4
    else:
        continue

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(稀勢の里:0 橋本環奈:1 広瀬すず:2)
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を100x100pixelに変換し、1要素が[R,G,B]3要素を含む配列の100x100の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath).resize((100, 100)))
            print(filepath)
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            image = image.transpose(2, 0, 1)
            print(image.shape)
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0,0], 1 -> [0,1,0] という感じ。
Y = to_categorical(label_list)

# モデルを生成してニューラルネットを構築
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 100, 100)))
model.add(Activation("relu"))
model.add(Convolution2D(32, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode=("same")))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

#model.add(Dense(3))
model.add(Dense(5))
model.add(Activation("softmax"))

# オプティマイザにAdamを使用
opt = Adam(lr=0.0001)
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
#model.fit(image_list, Y, nb_epoch=1000, batch_size=25, validation_split=0.1)
model.fit(image_list, Y, nb_epoch=200, batch_size=25, validation_split=0.1)

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("/home/pi/2018.03.20/"):
    if dir == ".DS_Store":
        continue

    dir1 = "/home/pi/2018.03.20/" + dir 
    label = 0

    if dir == "back":
        label = 0
    elif dir == "D-2":
        label = 1
    elif dir == "D-3":   # 橋本環奈はラベル1
        label = 2
    elif dir == "D-4":   # 橋本環奈はラベル1
        label = 3
    elif dir == "window":      # 広瀬すずはラベル2
        label = 4
    else:
        continue

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((100, 100)))
            print(filepath)
            image = image.transpose(2, 0, 1)
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")
