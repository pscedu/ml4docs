# ml4docs
Machine Learning for Docs projects developed in collaboration with Prof. Raja Adal at Pitt.

/pylon5/hm5fp1p/data/ is the location of the original 6000x4000 images. About 108 of them have human-annotated stamps in labelme format and the rest are unannotated. The annotated and unannotated images are mixed together in one image folder. There are a total of 389 jpeg files. The annotations are xml files. There are 126 xml files.

├── data

       ├── annotations
              └── *.xml (126 stamp annotations)
       |── images
             └── *.jpg (389 images, only 108 have annotations)

/pylon5/pscstaff/myilmaz/Stamps_Data is where I place large datasets I want to share with the group. This is where the predicted machine-annotated images (1800x1200) will be located in labelme format. They will be in folders labelled by Stage.


├── Stamps_Data

       |-- Stage1
          |--- predicted012320
       
              ├── annotations
                  └── *.xml (282 stamp annotations)
              |── images
                  └── *.jpg (282 images)
                  
                  
/pylon5/pscstaff/myilmaz/Stamps_Data/original_stamps_images is the location of the original 100 training images but resized to 1200x1800. There are no annotations in this folder, just images, although these images were human-annotated for stamps when they were a larger size (6000x4000). These images can be used for adding page annotations using LabelMe.
                  

