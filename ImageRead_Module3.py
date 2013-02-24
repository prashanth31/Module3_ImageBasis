"""
Prashanth Khambhammettu (prashanth31 at gmail dot com)
2013/02/24 : Python implementation of Image Basis Example
"""


import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from math import *



def haar(N) :
    """
    Function to calculate the Haar Basis
    Assume N is power of 2
    """

    h = np.zeros((N,N))
    h[0,:] = 1/sqrt(N)
        
    for k in range(1,N):
        p = int(log(k)/log(2)) 
        q = k-(2**p)
        k1 = 2**p
        t1 = N/k1 
        k2 = 2**(p+1)
        t2 = N/k2
        for i in range(1,t2+1) : 
            h[k+1-1,i+q*t1-1] = (2.0**(p/2.0))/sqrt(N)
            h[k+1-1,i+q*t1+t2-1] = -(2.0**(p/2.0))/sqrt(N)
    
    return h        
        

if __name__ == "__main__":
    #Load image
    im =Image.open('camera.jpg')
    #Convert the Image to a numpy matrix
    I =np.array(im)
    #Make a copy of the original image for ErrorNorm Calculation
    Iorig =I.copy()
    
    #Plot on the left side
    ax1 =plt.subplot(1,2,1)
    plt.imshow(I,cmap='gray',interpolation='none')
    
    #Let's flatten I so that it looks like the Matlab I
    I=I.flatten(1)
    
    # Generate Haar basis vector (rows of H)
    H = haar(4096)
    #
    # Project image on the new basis
    I_haar=np.dot(H,I)
    
    # Remove the second half of the coefficient
    I_haar[2048:4095]=0
    
    
    #Recover the image by inverting change of basis
    I_haar = np.dot(H.transpose(),I_haar)
    
    #Rearrange pixels of the image
    I_haar = (I_haar.transpose()).reshape(64,64,order='F').copy()
    
    #Plot Image on the right hand side
    ax2=plt.subplot(1,2,2)
    plt.imshow(I_haar,cmap='gray',interpolation='none')
    
    #Calculate Error matrix
    error=Iorig-I_haar
    distance = sqrt(np.sum(error*error))
    print 'Error distance =', distance
    #Show the Images
    plt.show()