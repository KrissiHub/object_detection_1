import os
#import OpenCV_Cluster as oc
import OpenCV_Cluster.get_diff
import OpenCV_Cluster.save_originals
import OpenCV_Cluster.cut
import OpenCV_Cluster.cluster
import OpenCV_Cluster.kmeans

import Yolov5_Cluster.cluster
import Yolov5_Cluster.delete_duplicates

#OPENCV_CLUSTER

#Define the variables that arent gonna change
ORIGINAL_FOLDER = "/Volumes/Seagate\ Expansion\ Drive/monstercam"
DIFF_FOLDER = "Difference_Images"
FILTERED_IMAGES_ORIGINAL = "Original_Images"
LABEL_FOLDER = "Label"
CROP_FOLDER = "Crop/"
CROP_FOLDER_SMALL = "Crop_Small"
CLUSTER_FOLDER = "Cluster03_big_nog"
LABEL_YOLO = "Yolov5_Cluster/Label"
CLUSTER_YOLO = "Yolo_Cluster"
CLUSTER_YOLO_CONF = "Yolo_Cluster_Conf"
CLUSTER_YOLO_CONF_DEL = "Yolo_Cluster_Conf_Del"
KMEANS_CLUSTER = "Kmeans_Cluster_5"
KMEANS_TEST = "object_detection_1/Test_Kmeans"
#the cropping information for the cut function
MIN_WIDTH = 50
MIN_HEIGHT = 40

#make all the neccesary directories
os.system("mkdir Difference_Images")
os.system("mkdir Original_Images")
os.system("mkdir Label")
os.system("mkdir Crop")
os.system("mkdir Crop_Small")
os.system("mkdir Cluster03_big_nog")
os.system("mkdir Yolov5_Cluster/Label")
os.system("mkdir Yolo_Cluster")
os.system("mkdir Yolo_Cluster_Conf")
os.system("mkdir Kmeans_Cluster_5")

#counter variables for the cluster function
I = 0
COUNTER = 0


#call the diff function -> we get the diff images, and also filter the huge amount of data we got

#OpenCV_Cluster.get_diff.load_images_from_read_folder(ORIGINAL_FOLDER, DIFF_FOLDER)

#call the save originals function -> we get the fitting now filtered originals of the diff images

#OpenCV_Cluster.save_originals.save_originals(DIFF_FOLDER, FILTERED_IMAGES_ORIGINAL, ORIGINAL_FOLDER)

#call the cut function -> detects and cuts possible insects out of the originals

#OpenCV_Cluster.cut.regions_of_interest(DIFF_FOLDER, FILTERED_IMAGES_ORIGINAL, LABEL_FOLDER, MIN_WIDTH, MIN_HEIGHT, CROP_FOLDER, CROP_FOLDER_SMALL)

#call the cluster funtcion -> clusters the cut images, so a human can look over it

#OpenCV_Cluster.cluster.cluster_images(COUNTER, I, CROP_FOLDER, CROP_FOLDER)
OpenCV_Cluster.kmeans.image_feature("5", KMEANS_CLUSTER)


#YOLOV5_CLUSTER

#call the filter function -> filters the images where there is something happening

#Yolov5_Cluster.cluster.load_and_run_yolo(FILTERED_IMAGES_ORIGINAL, LABEL_YOLO, CLUSTER_YOLO_CONF)

#calls the delete_dups function -> get rid of all the duplicates in our cluster
#Yolov5_Cluster.delete_duplicates.delete_dups(CLUSTER_YOLO_CONF, CLUSTER_YOLO_CONF_DEL)

#call the crop function -> crops the image so a human can look over it