import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

filePath = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\Images\\"

def load_data(fileName):
	x_img = []
	y_label = []
	
	file = open(fileName,"r") 
	data = file.readlines()
	for line in data:
		index = line.index('g')  
		imgName = line[0:index+1]
		label = line[index+2:index+7]
		img = filePath+imgName
		
		words = line

	fileName = words[0:9]
	label = words[10:16]
	print(filePath+fileName)
	print(label)
	
	img = filePath+fileName
	print(img)
	im = np.array(Image.open(img))
	# tmp = Image.open(img)
	plt.imshow(im)
	plt.axis('off')
	plt.show()
	
if __name__ == "__main__":    

	file = "D:\\code\\NotePad\\SCUT-FBP5500_v2\\train_test_files\\split_of_60%training and 40%testing\\train.txt"
	load_data(file)
	sStr1 = 'strchr'   
	sStr = 's'   
	nPos = sStr1.index(sStr)   
	print(nPos)
