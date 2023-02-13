import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import numpy as np
import cv2 as cv

#np.set_printoptions(threshold=os.sys.maxsize)

read_folder = "Test_Kmeans"

arr = None
for file in sorted(os.listdir(read_folder)):
    if file.startswith("."):
        continue
    image = cv.imread(read_folder + "/" + file)
    if arr is None:
        arr = np.array(image)
    else:
        arr = np.append(arr, image)
arr = arr.reshape(-1,1)
print(arr)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit_predict(arr)