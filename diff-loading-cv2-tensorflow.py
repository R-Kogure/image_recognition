#cv2.imreadとtensorflow.keras.preprocessing.image.load_dataの差
import cv2
import tensorflow
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

img_cv2 = cv2.imread("example.jpg")
img_tensorflow = tensorflow.keras.preprocessing.image.load_img("example.jpg")

"""
配列の差

cv2で読み込むと (height, width, channels) の三次元numpy配列として読み込まれる

tensorflowで読み込むと (samples, height, width, channnels) の四次元numpy配列として読み込まれる

"""

#cv2で読み込んだ画像をtensorflowで扱えるようにするために

h, w = 224, 224

#resize
img_cv2 = cv2.resize(img_cv2, (h,w))

#d-type conversion
img_cv2 = img_cv2.astype("flowt32") #任意

#標準偏差で割る
img_cv2 /= np.std(img_cv2, axis =0)

#チャンネルの次元を入れ替えてtensorflowで扱えるようにする : img_to_array
new_img = np.expand_dims(img_to_array(img_cv2), axis =0)



#author : R. Kogure