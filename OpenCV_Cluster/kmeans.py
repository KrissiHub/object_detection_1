from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras import utils
from keras.utils import img_to_array
from keras.utils import load_img
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
from PIL import Image 
import cv2 as cv
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import datasets
#np.set_printoptions(threshold=os.sys.maxsize)


# Function to Extract features from the images
def image_feature(read_folder, write_folder):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = []
    img_name = []
    im_read = []
    for file in os.listdir(read_folder):
        if file.startswith("."):
            continue
        fname=read_folder+'/'+file
        im_read.append(cv.imread(fname))
        img=utils.load_img(fname,target_size=(224,224))
        x = img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(file)
    kmeans(features, img_name, write_folder, read_folder, im_read)

#img_path=os.listdir('cluster')
#img_features,img_name=image_feature(img_path)

def kmeans(features, img_name, write_folder, read_folder, im_read):
    k=6
    clusters = KMeans(k, random_state = 40)
    clusters.fit(features)
    #sil_vis(clusters, features)   
    #saves the labels in a panda data frame for better sorting
    image_cluster = pd.DataFrame(img_name ,columns=['image'])
    image_cluster["clusterid"] = clusters.labels_
   

    sort_to_dir(image_cluster, write_folder, read_folder)


def sort_to_dir(image_cluster, write_folder, read_folder):
    image_cluster = image_cluster.reset_index()
    for index, row in image_cluster.iterrows():
        cluster_id = str(row["clusterid"])
        image_name_sol = str(row["image"])
        image_name= cv.imread(read_folder + "/" + image_name_sol)
        if os.path.exists(write_folder+"/"+cluster_id):
            cv.imwrite(write_folder+"/"+cluster_id+"/"+image_name_sol, image_name)
        else:
            os.mkdir(write_folder+"/"+cluster_id)
            cv.imwrite(write_folder+"/"+cluster_id+"/"+image_name_sol, image_name)


def sil_vis(clusters, features):
    fig, ax = plt.subplots(3, 3, figsize=(15,8))
    features = np.array(features)
    sil = []
    for i in range(7,12):
        clusters = KMeans(i, random_state = 40)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(clusters, colors='yellowbrick')
    
        sil.append(visualizer.fit(features))
        visualizer.show()



