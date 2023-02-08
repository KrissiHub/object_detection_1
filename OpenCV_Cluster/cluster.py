import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt 





cluster_dict = dict()
already_in_dict = False
files_list = []



#checks if an item is already a value or a key
def key_or_value(file, cluster_dict):
    is_value = False
    for value in cluster_dict.values():
            value_list = value.split()
            for item in value_list:
                #print(item)
                if item == file:
                    is_value = True
    print(file + " already in dict? " + str(is_value))
    return is_value


def cluster_images(counter, i, read_folder, read_folder_big):
    list_directory_read = sorted(os.listdir(read_folder))
    length = len([name for name in os.listdir(read_folder)])
    for filename in sorted(os.listdir(read_folder)):
        #skip hidden files
        if filename.startswith("."):
            continue
        read_image = cv.imread(read_folder + "/" + filename)
        #convert the image to RGB
        read_image = cv.cvtColor(read_image, cv.COLOR_BGR2GRAY)
        counter += 1
        i += 1
        if key_or_value(filename, cluster_dict):
                continue
        else:
            cluster_dict[counter] = filename
        #count i up, so we dont have to go through the whole directory again and again
        #starts where we already have been
        for k in range(i, length):
            print(k)
            filename_compare = list_directory_read[k]
            print(filename)
            print(filename_compare)
            if filename_compare.startswith("."):
                continue
            #dont compare the first picture to itself
            if filename_compare == filename:
                print("Is same file")
                continue
            #checks if the image is already in a cluster, so we dont sort it twice
            if key_or_value(filename_compare, cluster_dict):
                continue
            read_image_compare = cv.imread(read_folder + "/" + filename_compare)  
            read_image_compare = cv.cvtColor(read_image_compare, cv.COLOR_BGR2GRAY)
            #calculate the histograms
            hist1 = cv.calcHist(read_image, [0], None, [256], [0,256])
            hist2 = cv.calcHist(read_image_compare, [0], None, [256], [0,256])
            compare_value = (cv.compareHist(hist1, hist2,cv.HISTCMP_CORREL))
            print(compare_value)
            #if the similarity is big enough for the images to be the same
            if compare_value >= 0.2:
                cluster_dict[counter] +=  " " + filename_compare

    #write_cluster(cluster_dict, read_folder, read_folder_big)
    print(cluster_dict.items())
    return cluster_dict
        


def write_cluster(cluster_dict, read_folder, read_folder_big):
    for key in cluster_dict.keys():
        os_ = "mkdir " + "Cluster/Cluster_" + str(key)
        os.system(os_) 
        cluster_images = cluster_dict[key].split()
        for item in cluster_images:
            write_img = cv.imread(read_folder_big + item)
            cv.imwrite(("Cluster/Cluster_" + str(key) + "/" + item), write_img)






