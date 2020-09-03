#!/usr/bin/python
from PIL import Image
import os, sys

dir = "/MLStamps/long-tail/OpenLongTailRecognition-OLTR/OLTRDataset/OLTRDataset_1/campaign3to5_160"

def resize():
    for i in os.listdir(dir):
    #files = 0
        for file in os.listdir(os.path.join(dir,i)):
            #if os.path.isfile(path+item):
                im = Image.open(dir+"/"+i+"/"+file)
                print(file)
               # f, e = os.path.splitext(path+item)
                imResize = im.resize((160,160), Image.BILINEAR)
                imResize.save(dir+"/"+i+"/"+file, 'JPEG', quality=90)

resize()
