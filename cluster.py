import os
#import OpenCV_Cluster as oc
import OpenCV_Cluster.get_diff
import OpenCV_Cluster.save_originals
import OpenCV_Cluster.cut
import OpenCV_Cluster.cluster

import Yolov5_Cluster.filter

#OPENCV_CLUSTER

#Define the variables that arent gonna change
ORIGINAL_FOLDER = "/Volumes/Seagate\ Expansion\ Drive/monstercam"
DIFF_FOLDER = "Difference_Images"
FILTERED_IMAGES_ORIGINAL = "Original_Images"
LABEL_FOLDER = "Label"
CROP_FOLDER = "Crop/"
CROP_FOLDER_SMALL = "Crop_Small"
CLUSTER_FOLDER = "Cluster"
#the cropping information for the cut function
MIN_WIDTH = 50
MIN_HEIGHT = 40

#make all the neccesary directories
os.system("mkdir Difference_Images")
os.system("mkdir Original_Images")
os.system("mkdir Label")
os.system("mkdir Crop")
os.system("mkdir Crop_Small")
os.system("mkdir Cluster")

#counter variables for the cluster function
I = 0
COUNTER = 0


#call the diff function -> we get the diff images, and also filter the huge amount of data we got

OpenCV_Cluster.get_diff.load_images_from_read_folder(ORIGINAL_FOLDER, DIFF_FOLDER)

#call the save originals function -> we get the fitting now filtered originals of the diff images

OpenCV_Cluster.save_originals.save_originals(DIFF_FOLDER, FILTERED_IMAGES_ORIGINAL, ORIGINAL_FOLDER)

#call the cut function -> detects and cuts possible insects out of the originals

OpenCV_Cluster.cut.regions_of_interest(DIFF_FOLDER, FILTERED_IMAGES_ORIGINAL, LABEL_FOLDER, MIN_WIDTH, MIN_HEIGHT, CROP_FOLDER, CROP_FOLDER_SMALL)

#call the cluster funtcion -> clusters the cut images, so a human can look over it

OpenCV_Cluster.cluster.cluster_images(COUNTER, I, CROP_FOLDER_SMALL, CROP_FOLDER)


#YOLOV5_CLUSTER

#call the filter function -> filters the images where there is something happening

Yolov5_Cluster.filter.load_images_from_read_folder(ORIGINAL_FOLDER, DIFF_FOLDER)

#call the detect function -> detects the boxes where there could be an insect and also clusters them

#call the crop function -> crops the image so a human can look over it