# ml4docs
Machine Learning for Docs projects developed in collaboration with Prof. Raja Adal at Pitt.
HLS = Human labeled stamps
MLS = Machine/model labeled stamps
HLP = Human labeled pages
MLP = Machine/model labeled pages


# Mitsui Mi'ike Mine document images:
Contains: JPEG Images 6000x4000, 70 folders

A link to the location where Raja is storing the full set of Mitsui Mi'ike Mine document images is below. There are 70 folders. We are using some files from folder 66 for training the model.
https://pitt-my.sharepoint.com/personal/rajaadal_pitt_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frajaadal%5Fpitt%5Fedu%2FDocuments%2F2%20Typewriter%2FTypewriter%20Japan%2FMitsui%20Project%2FData%2F%E4%B8%89%E6%B1%A0%E9%89%B1%E6%A5%AD%E6%89%80%E8%B3%87%E6%96%99&originalPath=aHR0cHM6Ly9waXR0LW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL3JhamFhZGFsX3BpdHRfZWR1L0VyQkpwWTdMVjdWT2xWTjliQ1luSkxnQnFQVTQ3SmtnRFJ2Wmo3VlVPMlNBd0E_cnRpbWU9NHI5NXM2cXAxMGc


# Our Original Dataset: 
Contains: 389 JPEG Images 6000x4000
Contains: 126 XML HLS Annotations for 108 image files

/pylon5/hm5fp1p/data/ is the location of the original 6000x4000 images we saved on Pylon5 to use for training the model. About 108 of them have human-annotated stamps in labelme format and the rest are unannotated. The annotated and unannotated images are mixed together in one image folder. There are a total of 388 jpeg files. The annotations are xml files. There are 126 XMl files.

/pylon5/pscstaff/myilmaz/Stamps_Data/labelme Is the above dataset in 1800x1200.

├── data

       ├── annotations
              └── *.xml (126 stamp annotations)
       |── images
             └── *.jpg (389 images, only 108 have annotations)

/pylon5/pscstaff/myilmaz/Stamps_Data This is where the predicted machine-annotated images (1800x1200) will be located in Labelme format (xml annotations). They will be in folders labelled by Stage.


├── Stamps_Data

       |-- stage1
          |--- predicted012320
       
              ├── annotations
                  └── *.xml (282 stamp annotations)
              |── images
                  └── *.jpg (282 images)
                  
                  
/pylon5/pscstaff/myilmaz/Stamps_Data/stamps_images-1800x1200 is the location of the original 100 training images but resized to 1200x1800. All of these images have annotations.There are no annotations in this folder, just images, but these images were human-annotated for stamps so annotations are available in the 'labelme' folder above. These images can be used for adding page annotations using LabelMe.
                  
/pylon5/pscstaff/myilmaz/Stamps_Data/stage2/selection These are the new machine-annotated stamps images (1800x1200) that have been selected to have stamps corrected and page annotations added using Labelme. They will be used for training data in Stage 2. They have new names. They were produced by training the model 10 epochs with Keras-Retinanet at batch size of 2 and 1000 steps. The rest were defaults.
