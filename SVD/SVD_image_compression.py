# -*- coding: utf-8 -*-
"""
The code aims to read any high resolution B/W Image and do its SVD 
in order to find a compressed representation without compromising much on image quality

@author: Suniti
"""

import numpy as np
from scipy import linalg as la
from numpy.linalg import multi_dot
from PIL import Image


#Reading JPG image 

img1=Image.open("D:/SVD/SVD4.jpg")


Rimg1=np.array(img1)/255 #Scaling the image 
orig_img=np.array(img1)
len,wid=orig_img.shape

#Image.fromarray(np.array(img1)).show()


R_T_R=Rimg1.T.dot(Rimg1)


Rlambda,Reigvec=la.eig(R_T_R) #finding the spectral decomp. of RtR



Rlambda=Rlambda.real
idx=np.argsort(Rlambda)[::-1]
Rlambda=Rlambda[idx]
Rsing=np.abs(np.sqrt(np.abs(Rlambda)))
D=np.diag(Rsing) #Constructing the diagonal matrix
Reigvec1=Reigvec[:,idx].real #V matrix of right singular vectors 
U=Rimg1.dot(Reigvec1)
for i in range(Rlambda.shape[0]): #Construction the U matrix of left singular vectors
    U[:,i]=U[:,i]/Rsing[i]




k=min(150, len,wid) #choosing the no. of singular values for compression
B=multi_dot([U[:,0:k],D[0:k,0:k],Reigvec1.T[0:k,:]])*255  #SVD reconstruction with k nodes


MSEI=np.sum(np.square(orig_img-B))/(len*wid)
MaxI=np.max(orig_img).astype('int32')
temp=np.square(MaxI)/MSEI
PSNR=10*np.log10(temp) #computing Peak Signal to Noise Ratio
print("The MSE for image compression is :"+np.str(MSEI))
print("The max pixel value for image is :"+np.str(MaxI))
print("The Peak signal to noise ratio for compression is "+np.str(PSNR))


Noiseimg=np.abs(orig_img-B)

Image.fromarray(B).show() #viewing the compressed reconstructed image

Image.fromarray(Noiseimg).show() #viewing the Noise part of the image that is left over


    


