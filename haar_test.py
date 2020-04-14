import haarlikefeature
import numpy as np
import timeit
import cv2
import sys
from test import extract_feature_image
import  matplotlib.pyplot as plt
import pywt
import pywt._thresholding as thres
# np.set_printoptions(threshold=sys.maxsize)
img_size=256
path="C:/Users/kk/Desktop/datatest-001/"

img=cv2.imread(path+"2.png",0)
img=cv2.resize(img,(img_size,img_size))
# cv2.imshow("origin",img)
# haarlike=haarlikefeature.HaarlikeFeature()

start = timeit.default_timer()
cof=pywt.wavedec2(img,'haar')
arr,slicer=pywt.coeffs_to_array(cof)
res = pywt.waverec2(cof,'haar')
print(arr)

# features_cnt, descriptions = haarlike.determineFeatures(img_size, img_size)
# features_descriptions = descriptions[::-1]
# features = haarlike.extractFeatures(img, features_descriptions)
# print(features.shape)
stop1 = timeit.default_timer()
# features=np.reshape(features,(-1,16))
# cv2.imshow("vio",features)
#
#print(arr)

print('Time 1 : ', stop1 - start)
# res=extract_feature_image(img,img_size)
# print(res.shape)
# cv2.imshow("ress",res)

stop2 = timeit.default_timer()
print('Time 2 : ', stop2 - stop1)

cv2.waitKey(0)
cv2.destroyAllWindows()