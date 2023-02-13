import os
import torch
import pandas
import cv2 as cv

def load_and_run_yolo(read_folder, label_folder):
    #move the raw images to the new yolov5 directory
    #CHANGE THE PATH SO IT CAN BE MODIFIED BETTER
    #os_ = "rsync --exclude=".*" /Volumes/WD_BLACK/Algorithm/" + read_folder + " /Volumes/WD_BLACK/Algorithm/object_detection_1/Yolov5_Cluster/yolov5"
    #os.system(os_)



    #first we load the model we created (our pretrained algorithm model)
    model = torch.hub.load('/Volumes/WD_Black/Algorithm/object_detection_1/Yolov5_Cluster/yolov5', 'custom', path='/Volumes/WD_Black/Algorithm/object_detection_1/Yolov5_Cluster/yolov5/runs/train/exp/weights/best.pt', source='local')


    #we run through the whole original images folder to run the trained model over every single image and detect the objects
    for image in sorted(os.listdir("/Volumes/WD_BLACK/Algorithm/object_detection_1/Yolov5_Cluster/yolov5/Original_Images")):
        if image.startswith("."):
            continue
        image_name = os.path.basename(os.path.normpath(image))
        print(image_name)
        image = "/Volumes/WD_Black/Algorithm/object_detection_1/Yolov5_Cluster/yolov5/Original_Images/" + image
        result = model(image)
        #to get the cropped result
        #SOMETHING IS NOT RIGHT HERE
        crop = result.crop(save=True)
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
        #print(boundin_box_information)
        break
        