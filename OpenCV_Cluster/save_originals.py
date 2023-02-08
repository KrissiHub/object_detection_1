import cv2 as cv
import os
from os import listdir


def save_originals(read_folder, write_folder, original_folder):
    for filename in sorted(os.listdir(read_folder)):
        if filename.startswith("."):
            continue
        split_append = filename.split("%")
        original_name = original_folder + "/" + split_append[0] + "/" + split_append[1]
        os.system("cp " + original_name + " " + write_folder)
        print(original_name)