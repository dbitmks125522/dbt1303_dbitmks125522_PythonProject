import numpy as np
import cv2 #This is how we import the OpenCV library

#include the following since we are running on Google Colab
from google.colab.patches import cv2_imshow #This will help in displaying the image as we continue to modify it
from google.colab import files #This will help us to select any image from our local files for editing

#create a function that loads the image:
def readFile(file_name)
    image=cv2.imread(file_name)
    cv2_imshow(image)
    return image


 #Call the function to load the image:
 uploaded=files.upload()
file_name=next(iter(uploaded))
img=readFile(file_name)

#We use the edge mask to emphasize the thickness of the image’s edges.
def edgeMask(image,lineSize,blurValue)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayBlur=cv2.medianBlur(gray,blurValue)
    edges=cv2.adaptiveThreshold(grayBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,CV2.THRESH_BINARY,lineSize,blurValue)
    return edges

#Call the defined function
lineSize=7
blurValue=7
edges=edgeMask(image,lineSize,blurValue)
cv2_imshow(edges)

#Define a color quantization function
def colorQuantization(image,k)
    data=np.float32(image).reshape((-1,3))
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    result=center[label.flatten()]
    result=result.reshape(img.shape)
    return result

#call the function
colors=9
image=colorQuantization(image,colors)

#The image noise is reduced using this code
blurImage=cv2.bilateralFilter(image,d=7,sigmaColor=200,sigmaSpace=200)

#Finally,combine “color-quantized image” with the edge mask using the cv2.bitwise_and() method.
cartoonImage=cv2.bitwise_and(blurImage,blurImage,mask=edges)