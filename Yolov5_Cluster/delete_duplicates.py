import os
import cv2 as cv
import numpy as np

def delete_dups(read_directory, write_directory):
    images_to_compare = []
    for dirs in sorted(os.listdir(read_directory)):
        if dirs.startswith("."):
            continue
        for dir in sorted(os.listdir(read_directory + "/" + dirs)):
            if dir.startswith("."):
                continue
            tail = None
            for file in sorted(os.listdir(read_directory + "/" + dirs + "/" + dir)):
                if file.startswith("."):
                    continue
                if tail is None:
                    tail = read_directory + "/" + dirs + "/" + dir + "/" + file
                    continue
                current_image = cv.imread(read_directory + "/" + dirs + "/" + dir + "/" + file)
                tail_image = cv.imread(tail)
                if current_image.shape != tail_image.shape:
                    #tail is now the current image
                    tail = read_directory + "/" + dirs + "/" + dir + "/" + file
                    continue
                difference = current_image.copy()
                print(current_image.shape)
                print(tail_image.shape)
                cv.absdiff(current_image, tail_image, difference)
                diff = np.sum(difference)
                if diff == 0:
                    os.remove(tail)
                tail = read_directory + "/" + dirs + "/" + dir + "/" + file
            tail = None
            
        








                
            
        