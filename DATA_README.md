# ml4docs
Machine Learning for Docs projects developed in collaboration with Prof. Raja Adal at Pitt.

/pylon5/hm5fp1p/data/ is the location of the original 6000x4000 images. About 108 of them have human-annotated stamps in labelme format and the rest are unannotated. The annotated and unannotated images are mixed together in one image folder. There are a total of 389 jpeg files. The annotations are xml files. There are 126 xml files.

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
                  
/pylon5/pscstaff/myilmaz/Stamps_Data/stage2/selection These are the new machine-annotated stamps images (1800x1200) that have been selected to have stamps corrected and page annotations added using Labelme. They will be used for training data in Stage 2.
