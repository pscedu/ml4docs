#!/usr/bin/python
from PIL import Image
import os, sys

dir = "/MLStamps/long-tail/OpenLongTailRecognition-OLTR/combined-200-stamps-6Kx4K-croppedStamps"

def resize():
    count = 0
    for i in os.listdir(dir):
    #files = 0
        #for file in os.listdir(os.path.join(dir,i)):
            #if os.path.isfile(path+item):
        im = Image.open(dir+"/"+i)
        print(i)
        print(count)
        count += 1
               # f, e = os.path.splitext(path+item)
        imResize = im.resize((256,256), Image.BILINEAR)
        imResize.save(dir+"/"+i, 'JPEG', quality=90)

resize()
