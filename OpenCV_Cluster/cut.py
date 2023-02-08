import cv2 as cv
from os import listdir
import os
import numpy as np
from numpy.linalg import norm

#sets a brightness value, so we can differanciate between dark and bright images
MEAN_BRIGHTNESS = 10
#the original image name and the bounding box information will be saved in here
label_dictionary = dict()
#size of the rectangles depending on the brightness
MAX_WIDTH_DARK = 150
MAX_HEIGHT_DARK = 65
MAX_WIDTH_BRIGHT = 150
MAX_HEIGHT_BRIGHT = 90

#puts filter on the image, so that our region of intrest is in the foreground
def regions_of_interest(read_folder, original_image_folder, label_folder, min_width, min_height, crop_folder, crop_folder_small):
	#so we now if its a dark or a light image
	is_dark = False
	counter = 0
	for file in sorted(os.listdir(read_folder)):
		if file.startswith("."):
			continue
		
		original_image = cv.imread(read_folder + "/" + file)
		#first we create a mask, so we can concentrate on the important ares
		original_image = cut_abundance(original_image)
		#calculate the average brighntess of the image
		brighntess = np.average(norm(original_image, axis=2)) / np.sqrt(3)
		#depending on the brighntess of the image, we use two different segmentation models
		if brighntess <= MEAN_BRIGHTNESS:
			is_dark = True
			#make the image brighter so we can find all the insects
			brighter_image = cv.convertScaleAbs(original_image, 1, 10)
			#now we convert the RGB image to gray scale
			gray_image = cv.cvtColor(brighter_image, cv.COLOR_BGR2GRAY)
			#now we blur the image
			blur_image = cv.GaussianBlur(gray_image, (99, 99), 0)
			#threshhold to find the brightest spots (insects)
			thresh, thresh_image = cv.threshold(blur_image, 150, 255, cv.THRESH_BINARY)
			#now we find the contours of the highlighted areas
			contours, hirarchy = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			for contour in contours:
				(x,y,w,h) = cv.boundingRect(contour)
				
				cv.rectangle(thresh_image, (x,y), (x+w, y+h), (255,255,255), 4)
				bounding_box = {"x": x, "y": y, "w": w, "h" : h}
				counter += 1
				cut_and_save(bounding_box, file, is_dark, counter, original_image_folder, label_folder, crop_folder, crop_folder_small)
				
			
		elif brighntess > MEAN_BRIGHTNESS:
			is_dark = False
			#now we convert the RGB image to gray scale
			gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
			#now we blur the image
			blur_image = cv.GaussianBlur(gray_image, (5, 5 ), 0)
			#threshhold to find the brightest spots (insects)
			thresh, thresh_image = cv.threshold(blur_image, 50, 255, cv.THRESH_BINARY)
			#save the contours 
			contours, hirarchy = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			for contour in contours:
				(x,y,w,h) = cv.boundingRect(contour)
				cv.rectangle(thresh_image, (x,y), (x+w, y+h), (255,255,255), 4)
				bound = "w" + str(w) + "h" + str(h)
				bounding_box = {"x": x, "y": y, "w": w, "h" : h}
				counter += 1
				cut_and_save(bounding_box, file, is_dark, counter, original_image_folder, label_folder, min_width, min_height, crop_folder, crop_folder_small)
		

	

#cuts the grass part and the time stamps out of the image
def cut_abundance(image_to_be_circled):
	#first we need to get the center coordinates of the picture
	(h,w) = image_to_be_circled.shape[:2]
	h -= 450
	w -= 200

	#we save the original height and width from the image
	#now create the default mask
	mask = np.zeros_like(image_to_be_circled)
	mask = cv.circle(mask, (w//2, h//2), 1850, (255,255,255), -1)
	#subtracts the mask and the original image so the abundance is now gone
	mask_out = cv.subtract(mask, image_to_be_circled)
	mask_out = cv.subtract(mask, mask_out)
	return mask_out



#gets passed the bounding box information, cuts from the original image, saves the org image name and the bb information in a seperate txt
def cut_and_save(bounding_box_info, image, is_dark, img_name_counter, original_image_folder, label_folder, MIN_WIDTH, MIN_HEIGHT, crop_folder, crop_folder_small):
	#check if the bounding box is the right size to be an insect depending on the brightness
	#for dark images
	if is_dark:
		if (bounding_box_info["w"] <= MAX_WIDTH_DARK) and (bounding_box_info["h"] <= MAX_HEIGHT_DARK):
			if (bounding_box_info["w"] >= MIN_WIDTH) and (bounding_box_info["h"] >= MIN_HEIGHT):
				#look for the original image  
				split_append = image.split("%")
				#this is the name of the original file and place
				original_image_name = original_image_folder + "/" + split_append[1]
				#safe the bounding box info as a string to save in the txt
				bounding_string = str(bounding_box_info["x"]) + " " + str(bounding_box_info["y"]) + " " + str(bounding_box_info["w"]) + " " + str(bounding_box_info["h"])
				#now we crop the image
				org_img = cv.imread(original_image_name)
				if org_img is None:
					print("none")
				print(original_image_name)
				crop_info = define_ractangle(org_img, bounding_box_info)
				crop_small = org_img[bounding_box_info["y"]:bounding_box_info["y"]+bounding_box_info["h"], bounding_box_info["x"]:bounding_box_info["x"]+bounding_box_info["w"]]
				crop = org_img[crop_info["y"]:crop_info["y"]+crop_info["h"]+200, crop_info["x"]:crop_info["x"]+crop_info["w"]+200]
				cut_name = crop_folder + image
				cut_name_small = crop_folder_small + image
				#check if file already exists, so we dont write over it
				if os.path.isfile(crop_folder + image):
					image_without_extension = os.path.splitext(image)[0]
					cut_name = crop_folder + image_without_extension + str(img_name_counter) + ".jpg"
					cut_name_small = crop_folder_small + image_without_extension + str(img_name_counter) + ".jpg"
					cv.imwrite(cut_name, crop)
					cv.imwrite(cut_name_small, crop_small)
				else:
					cv.imwrite(crop_folder + image, crop)
					cv.imwrite(crop_folder_small + image, crop_small)
				#check if the txt doesnt already exits and if just add to it
				if os.path.isfile(label_folder + "/" + image + ".txt"):
					with open(label_folder + "/" + image + ".txt", "a") as f:
						f.write("\n" + cut_name + " " + bounding_string)
				#now we save the bounding box information and the title of the file in a txt file
				else:
					with open(label_folder + "/" + image + ".txt", "w") as f:
						f.write(cut_name + " " + bounding_string)
	elif not is_dark:
		if (bounding_box_info["w"] <= MAX_WIDTH_DARK) and (bounding_box_info["h"] <= MAX_HEIGHT_DARK):
			if (bounding_box_info["w"] >= MIN_WIDTH) and (bounding_box_info["h"] >= MIN_HEIGHT):
				#look for the original image  
				split_append = image.split("%")
				#this is the name of the original file and place
				original_image_name = original_image_folder + "/" + split_append[1]
				#safe the bounding box info as a string to save in the txt
				bounding_string = str(bounding_box_info["x"]) + " " + str(bounding_box_info["y"]) + " " + str(bounding_box_info["w"]) + " " + str(bounding_box_info["h"])
				#now we crop the image
				org_img = cv.imread(original_image_name)
				if org_img is None:
					print("none")
				print(original_image_name)
				crop_info = define_ractangle(org_img, bounding_box_info)
				crop_small = org_img[bounding_box_info["y"]:bounding_box_info["y"]+bounding_box_info["h"], bounding_box_info["x"]:bounding_box_info["x"]+bounding_box_info["w"]]
				crop = org_img[crop_info["y"]:crop_info["y"]+crop_info["h"]+200, crop_info["x"]:crop_info["x"]+crop_info["w"]+200]
				cut_name = crop_folder + image
				cut_name_small = crop_folder_small + image
				#check if file already exists, so we dont write over it
				if os.path.isfile(crop_folder + image):
					image_without_extension = os.path.splitext(image)[0]
					cut_name = crop_folder + image_without_extension + str(img_name_counter) + ".jpg"
					cut_name_small = crop_folder_small + image_without_extension + str(img_name_counter) + ".jpg"
					cv.imwrite(cut_name, crop)
					cv.imwrite(cut_name_small, crop_small)
				else:
					cv.imwrite(crop_folder + image, crop)
					cv.imwrite(crop_folder_small + image, crop_small)
				#check if the txt doesnt already exits and if just add to it
				if os.path.isfile(label_folder + "/" + image + ".txt"):
					with open(label_folder + "/" + image + ".txt", "a") as f:
						f.write("\n" + cut_name + " " + bounding_string)
				#now we save the bounding box information and the title of the file in a txt file
				else:
					with open(label_folder + "/" + image + ".txt", "w") as f:
						f.write(cut_name + " " + bounding_string)


#if the range is to big for the image
def define_ractangle(image, bounding_box_infos):
	x_crop = bounding_box_infos["x"] - 100
	y_crop = bounding_box_infos["y"] - 100
	w_crop = bounding_box_infos["w"] 
	h_crop = bounding_box_infos["h"] 
	yh_crop = y_crop + h_crop + 100
	xw_crop = w_crop + h_crop + 100
	#gets the range of the image
	(h_image, w_image) = image.shape[:2]

	#check if one of the now bigger coordinates is to big
	#check if x is out of range
	if x_crop < 0:
		x_crop = 0
	#check if y is out of range
	if y_crop < 0:
		y_crop = 0
	#check if yh is out of range
	if yh_crop > w_image:
		y_crop = w_image
	#check if wx is out of range
	if xw_crop > h_image:
		x_crop = h_image
	bigger_bb_info = {"y" : y_crop, "h" : h_crop, "x": x_crop, "w": w_crop}
	
	print((h_image, w_image))
	print(bounding_box_infos)
	return bigger_bb_info



