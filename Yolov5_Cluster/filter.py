import cv2 as cv
import sys
import os
import numpy as np



#image name if for naming and saving the filtered files, so they dont all have the same name
#image compare safes the two comparable images and is later passed on to the compare_to_images def
def load_images_from_read_folder(read_folder, write_folder_name):
    image_name = 0
    image_compare = []
    filename_test = []
    print("Start computing")
    #the directory is sorted, because otherwise the timestamps are to wide apart to compare
    for directory in sorted(os.listdir(read_folder)):
        if directory.startswith("."):
            continue
        print(directory)
        #saves org name for writing purposes
        original_directory = directory
        directory = read_folder + directory
        print(directory)
        for filename in sorted(os.listdir(read_folder)):
                #to not iterate over hidden files
                if filename.startswith('.'):
                    continue
                #save or filename for write purpose
                original_filename = filename
                filename = read_folder + "/" + filename
                print(filename)
                img = cv.imread(filename)
                #checks if image is good and we still have space
                if img is not None:
                    filename_test.append(filename)
                    image_compare.append(img)
                    print(filename)
                    #call the compare function and empty the images_compare array for new images, only if we have 2 images to compare
                    if len(image_compare) == 2:
                        image_compare = compare_two_images(image_compare, read_folder, original_filename, filename_test, write_folder_name)
                        filename_test[0] = filename_test[1]
                        filename_test.pop()
                else:
                    print("Image is none " + filename)
      
    print("End of computing")
    return image_compare

#compares the recent image with the default one and if the difference is big enough, saves it
def compare_two_images(image_compare, now_directory, filename, test, write_folder_name):
    #copy the initial image image so the new one has the same / right size
    difference = image_compare[1].copy()
    #puts the difference of the two images in difference
    print(str(test))
    cv.absdiff(image_compare[0], image_compare[1], difference)
    dif = np.sum(difference)
    #if the difference is bigger than the value, it is big enough to safe it to the new filtered folter
    if dif > 100000000:
        print(dif)
        #names the new file, so we can trace it back to the original
        new_file_name = str(write_folder_name) + "%" + str(filename)
        print(new_file_name)
        #now does not save the difference image but the original, becaue my yolo is trained on the original images
        cv.imwrite(os.path.join(write_folder_name, new_file_name), image_compare[0])
    #takes the second image and puts it as the first one, so it can be compared to the next
    image_compare[0] = image_compare[1]
    image_compare.pop()

    return image_compare





