import os
from PIL import Image
from numpy import *

def load_data(path1, path2, path3):
	#读入文件列表，path代表三个类别的文件路径    
	filelist1 = [os.path.join(path1, f) for f in os.listdir(path1)]    
	filelist2 = [os.path.join(path2, f) for f in os.listdir(path2)]    
	filelist3 = [os.path.join(path3, f) for f in os.listdir(path3)]  
	  
	x_test = []   
	y_test = []    
	n1 = len(filelist1)    
	n2 = len(filelist2)    
	n3 = len(filelist3)    
	for img in filelist1:        
		im = array(Image.open(img))
		#将图像格式转为numpy数组格式        
		# im = im.flatten()        
		x_test.append(im)    
	for img in filelist2:        
		im = Image.open(img)        
		# im = im.flatten()        
		x_test.append(im)    
	for img in filelist3:        
		im = array(Image.open(img))        
		# im = im.flatten()        
		x_test.append(im)    
	x_test = array(x_test)
		#自己造标签 总共三类，所以标签是012    
	y_test = zeros((n1 + n2 + n3), dtype=int)    
	for i in range(n1):         
		y_test[i] = 0    
	for i in range(n2):        
		y_test[n1 + i] = 1    
	for i in range(n2):        
		y_test[n1 + n2 + i] = 2    
	return x_test, y_test
	
	 
if __name__ == "__main__":    
x_test, y_test = load_data("E:\\xudata1\\test\\type0_1", "E:\\xudata1\\test\\type1_1",                               
"E:\\xudata1\\test\\type2_1")
