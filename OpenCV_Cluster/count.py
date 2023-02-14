import os
import cv2 as cv

'''counter = 0
for dir in sorted(os.listdir("/Volumes/WD_BLACK/Algorithm/Cluster")):
    if dir.startswith("."):
        continue
    for file in sorted(os.listdir("/Volumes/WD_BLACK/Algorithm/Cluster/" + dir)):
        counter +=1

print(counter)'''

'''list = ["Cluster_1", "Cluster_3", "Cluster_131"]
for dir in list:
    for file in os.listdir("/Volumes/WD_BLACK/Algorithm/Cluster03_big_nog/" + dir):
        counter +=1
print(counter
)'''
cpt = len([file for r, d, files in os.walk("/Volumes/WD_BLACK/Algorithm/object_detection_1/Yolo_Cluster_Conf") for file in files if not file.startswith(".")])
print(cpt)
