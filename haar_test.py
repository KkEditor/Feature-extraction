
import numpy as np
import timeit
import cv2
import sys
import  matplotlib.pyplot as plt
import pywt
import pywt._thresholding as thres
import scipy
from collections import Counter
# def calculate_entropy(list_values):
#     tup=hash(tuple(list_values))
#     counter_values = Counter(list_values).most_common()
#     probabilities = [elem[1]/len(list_values) for elem in counter_values]
#     entropy=scipy.stats.entropy(probabilities)
#     return entropy
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) >np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return crossings + statistics


# np.set_printoptions(threshold=sys.maxsize)
img_size=256
path="C:/Users/kk/Desktop/datatest-001/"

img=cv2.imread(path+"2.png",0)
img=cv2.resize(img,(img_size,img_size))
# cv2.imshow("origin",img)
# haarlike=haarlikefeature.HaarlikeFeature()

start = timeit.default_timer()
cof=pywt.wavedec2(img,'haar') #(len(cof)-1)/3
print(len(cof))
out = [item for t in cof for item in t]
# arr,slicer=pywt.coeffs_to_array(cof)
print(len(out))
# test=out[1]

for e in out:
    print(get_features(e))

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

# stop2 = timeit.default_timer()
# print('Time 2 : ', stop2 - stop1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()