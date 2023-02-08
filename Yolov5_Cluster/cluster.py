import os

#
def load_and_run_yolo():
    #install yolov5 with requirements, note: this is a forked version on my github, to modify te changes we need
    os.system("git clone https://github.com/KrissiHub/yolov5.git")
    os.system("cd yolov5")
    os.system("pip3 install -r yolov5/requirements.txt")

    #move the raw images to the new yolov5 directory
    os.system("cp -r Difference_Images yolov5")


    #algorithm detects objects, labels them and puts them sorted in direcotries, (in runs/exp)
    os.system("python3 yolov5/detect.py --weights yolov5s.pt --save-crop --save-txt --source yolov5/Images_Insects")

    #now the clustered directories should be copied from inside the yolo folder to 5 directories above, so they are not in the yolo algorithm anymore
    #the user doesnt have to fumble inside this folder

    os.system("mv yolov5/runs/detect/exp/crops Cluster")        