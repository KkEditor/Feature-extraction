import numpy as np
import timeit
import cv2
import pywt
import math
import os
import scipy.stats as stats
def von_neumann_entropy(density_matrix, cutoff=10):
    x = np.mat(density_matrix)
    one = np.identity(x.shape[0])
    base = one - x
    power = base*base
    result = np.trace(base)
    for k in range(2, cutoff):
        result -= np.trace(power) / (k*k - k)
        power = power.dot(base)

    # Twiddly hacky magic.
    a = cutoff
    for k in range(3):
        d = (a+1) / (4*a*(a-1))
        result -= np.trace(power) * d
        power = power.dot(power)
        result -= np.trace(power) * d
        a *= 2
    result -= np.trace(power) / (a-1) * 0.75
    return np.asarray(np.around((result / math.log(2)),2))

def cor(list_values):
    return np.corrcoef(list_values)

def calculate_statistics(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    mad=stats.median_absolute_deviation(list_values,axis=None)
    # print(mad)
    return n5, n25, median, mean, std, var, rms,no_mean_crossings,no_zero_crossings,mad


def get_features(list_values):
    statistics = calculate_statistics(list_values)
    entropy = von_neumann_entropy(list_values)
    return statistics

#use this func
#output: array of features
def haar_extract(img,size):
    img=cv2.resize(img,(size[0],size[1]))
    feature=[]
    cof=pywt.wavedec2(img,'haar')
    out = [item for t in cof for item in t]
    for e in out:
        feature.append(get_features(e))
    return np.ndarray.flatten(np.array(feature))


def main():
    start = timeit.default_timer()
    path="C:/Users/kk/Desktop/crop/"
    namelist=os.listdir(path)
    for i in namelist:
        img=cv2.imread(path+i,0)
        fea = haar_extract(img, (64, 64))
        print(fea)



    stop1 = timeit.default_timer()
    print("Time: ",stop1-start)

if __name__ == '__main__':
    main()