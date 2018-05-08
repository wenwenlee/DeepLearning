import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def load_data(textPath,ImgPath):
	x_img = []
	y_label = []
	
	file = open(textPath,"r") 
	data = file.readlines()
	for line in data:
		index = line.index('g')  
		imgName = line[0:index+1]
		label = line[index+2:index+7]
		img = ImgPath+imgName
		im = np.array(Image.open(img))
		x_img.append(im) 
		label_np = np.array(label)
		y_label.append(label_np)
		
	return x_img,y_label

if __name__ == "__main__":    

	textPath = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\train_test_files\\split_of_60%training and 40%testing\\train.txt"
	textPath_test = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\train_test_files\\split_of_60%training and 40%testing\\test.txt"
	ImgPath = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\Images\\"
	x_train,y_train = load_data(textPath,ImgPath)
	x_test,y_test = load_data(textPath_test,ImgPath)
