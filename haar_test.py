import numpy as np
import timeit
import cv2
import pywt


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

#use this func
#output: array of features
def haar_extract(img,size):
    img=cv2.resize(img,(size[0],size[1]))
    feature=[]
    cof=pywt.wavedec2(img,'haar')
    out = [item for t in cof for item in t]
    for e in out:
        feature.append(get_features(e))
    return feature

img_size=256
path="C:/Users/kk/Desktop/datatest-001/"

img=cv2.imread(path+"2.png",0)

start = timeit.default_timer()
fea=haar_extract(img,(256,256))
print(fea)

stop1 = timeit.default_timer()
print("Time: ",stop1-start)
