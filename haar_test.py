import numpy as np
import timeit
import cv2
import pywt
import math
import os
import scipy
import pandas as pd
from scipy.fftpack import dct
def median(list_values):
    return np.round(np.sum(np.nanmedian(list_values,axis=0)),6)
def von_neumann_entropy(density_matrix, cutoff=10):
    x = np.mat(density_matrix)
    one = np.identity(x.shape[0])
    base = one - x
    power = base*base
    result = np.trace(base)
    for k in range(2, cutoff):
        result -= np.trace(power) / (k*k - k)
        power = power.dot(base)
    result -= np.trace(power) / (cutoff - 1)
    return np.array(result / math.log(2),dtype=np.float64)

def calculate_statistics(list_values):
    # zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    # no_zero_crossings = len(zero_crossing_indices)
    # mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    # no_mean_crossings = len(mean_crossing_indices)
    # n5 = np.percentile(list_values, 5)

    #error
    # n25 = np.percentile(list_values, 25)
    # median = np.percentile(list_values, 50)
    # mad=np.array(stats.median_absolute_deviation(list_values,axis=None),dtype=np.float64)
    # coef=np.array(stats.variation(np.nanvar(list_values)))
    #dk why tho


    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    # var = np.nanvar(list_values)
    # rms = np.nanmean(np.sqrt(list_values ** 2))

    # return n5, n25, median, mean, std, var, rms,mad,coef
    return mean,std

def eigen_vector(list_values):
    W,v=np.linalg.eig(list_values)
    return np.round(np.real(np.sum(W)),6)

def get_features_dct(list_values):
    statistics = calculate_statistics(list_values)
    return statistics

def get_features(list_values):
    statistics = calculate_statistics(list_values)
    entropy = von_neumann_entropy(list_values)
    entropy=tuple(np.expand_dims(entropy,axis=0))
    W=eigen_vector(np.expand_dims(list_values,axis=0))
    W=tuple(np.expand_dims(W,axis=0))
    med=median(list_values)
    med = tuple(np.expand_dims(med, axis=0))
    return statistics + entropy + W +med



#use this func
#output: array of features
def haar_extract(img,size):
    img=cv2.resize(img,(size[0],size[1]))
    feature=[]
    cof=pywt.wavedec2(img,'haar')
    out = [item for t in cof for item in t]
    for e in out:
        sin=dct(img)
        feature.append(get_features_dct(sin))
        feature.append(get_features(e))
        # print(e.shape)
    # #level 1
    # cA1,cN1=pywt.dwt2(img,'haar')
    # sin1=dct(img)
    # feature.append(get_features(sin1))
    # for e in cN1:
    #     feature.append(get_features(e))
    #
    #
    # #level 2
    # cA2,cN2=pywt.dwt2(cA1,'haar')
    # sin2=dct(cA1)
    # feature.append(get_features(sin2))
    # for e in cN2:
    #     feature.append(get_features(e))
    #
    lst=list(sum(feature, ()))
    formated=["%.6f" % member for member in lst]
    return np.array(formated)


def main():
    start = timeit.default_timer()
    path="C:/Users/kk/Desktop/crop/"
    namelist=os.listdir(path)
    for i in namelist:
        img=cv2.imread(path+i,0)
        fea = haar_extract(img, (256, 256))
        print(fea)
        break

    stop1 = timeit.default_timer()
    print("Time: ",stop1-start)

if __name__ == '__main__':
    main()