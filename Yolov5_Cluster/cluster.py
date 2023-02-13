import os
import torch
import pandas
import cv2 as cv
import OpenCV_Cluster.cut
import math


def load_and_run_yolo(read_folder, label_folder, cluster):
    #move the raw images to the new yolov5 directory
    #CHANGE THE PATH SO IT CAN BE MODIFIED BETTER
    #os_ = "rsync --exclude=".*" /Volumes/WD_BLACK/Algorithm/" + read_folder + " /Volumes/WD_BLACK/Algorithm/object_detection_1/Yolov5_Cluster/yolov5"
    #os.system(os_)



    #save our work directory
    current_directory = os.path.abspath(os.getcwd())
    #first we load the model we created (our pretrained algorithm model)
    model = torch.hub.load(current_directory + '/Yolov5_Cluster/yolov5', 'custom', path=current_directory+'/Yolov5_Cluster/yolov5/runs/train/exp/weights/best.pt', source='local')


    #we run through the whole original images folder to run the trained model over every single image and detect the objects
    for image in sorted(os.listdir(current_directory+"/Yolov5_Cluster/yolov5/Original_Images")):
        if image.startswith("."):
            continue
        image_name = os.path.basename(os.path.normpath(image))
        print(image_name)
        image = current_directory+"/Yolov5_Cluster/yolov5/Original_Images/" + image
        image_read = cv.imread(image)
        #im_test = cv.imread(current_directory+"/Yolov5_Cluster/yolov5/Original_Images/""39501_monstercam_am_28-08-2022_um_06Uhr00.jpg")
        #cut the abundance, like the timestamps etc, so they dont get confused with objects of interest
        image_cut_abundance = OpenCV_Cluster.cut.cut_abundance(image_read, True)
        result = model(image_cut_abundance)
        #to get the cropped result
        #SOMETHING IS NOT RIGHT HERE
        crop = result.crop(save=False)
        #splice the label string of the crop, so we only have the name of the class, last 5 characters contain the confidence (and a space)
        #i.e Other Insect 0.70
        if crop:
            crop_conf = float(crop[0]["label"][-4:])
            #round the decimal, so we dont get to many clusters
            crop_conf = math.floor(crop_conf*10)/10
            crop_label = crop[0]["label"][:-5]
            crop_im = crop[0]["im"]
        #now we need to check if there is already a directory with the label and save it there
        if os.path.exists(current_directory+"/"+cluster+"/"+crop_label):
            if os.path.exists(current_directory+"/"+cluster+"/"+crop_label+"/"+str(crop_conf)):
                cv.imwrite(current_directory+"/"+cluster+"/"+crop_label+"/"+str(crop_conf)+"/"+image_name, crop_im)
            else:
                os.mkdir(current_directory+"/"+cluster+"/"+crop_label+"/"+str(crop_conf))
                cv.imwrite(current_directory+"/"+cluster+"/"+crop_label+"/"+str(crop_conf)+"/"+image_name, crop_im)

        #if not we make a new directory with the label name
        else:
            os.mkdir(current_directory+"/"+cluster+"/"+crop_label)
            os.mkdir(current_directory+"/"+cluster+"/"+crop_label+"/"+str(crop_conf))
            #print(current_directory+"/"+crop_label+"/"+image)
            cv.imwrite(current_directory+"/"+cluster+"/"+crop_label+"/"+str(crop_conf)+"/"+image_name, crop_im)


        #cv.imwrite("/Volumes/WD_Black/Algorithm/object_detection_1/Yolov5_Cluster/test.jpg", crop_im)
        #to extract the bounding box information from the cropped result
        panda = result.pandas().xyxy[0] 
        if panda.empty:
            continue
        print(panda)
        #has to be done i a for loop, because there could be multiple bounding boxes per image
        bounding_box_information = ""
        for i in range(0, panda.shape[0]):
            xmin = panda._get_value(i, 'xmin')
            ymin = panda._get_value(i, 'ymin')
            xmax = panda._get_value(i, 'xmax')
            ymax = panda._get_value(i, 'ymax')
            bounding_box_information += str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n"

        #with this information a label directory is filled with txt files, containg the image name and the bounding box information
        if os.path.isfile(label_folder + image_name + ".txt"):
            with open(label_folder + "/" + image_name + ".txt", "a") as f:
                f.write("\n" + bounding_box_information)
				#now we save the bounding box information and the title of the file in a txt file
        else:
            with open(label_folder + "/" + image_name + ".txt", "w") as f:
                f.write(bounding_box_information)
        
        
        