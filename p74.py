import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

def getdata():
    (x_train,t_train),(x_test,t_test)=\
        load_mnist(normalize=False,flatten=True)
    return x_train,t_train

x,t=getdata()
img=x[10]
label=t[10]
print(label)#5

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)
