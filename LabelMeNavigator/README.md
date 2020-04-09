Initialize the env for modifications
    ssh vm041.bridges.psc.edu
    source /opt/virtualenvs/navigator/bin/activate
    cd /opt/repositories/navigator/

Stop the running service
    screen -R
    Ctrl + C
    Ctrl + a, d

Modify the management command for setting the new stage as default
    vim labelMeNav/views.py +39
        models.Image.objects.create(image_name=item, stage__name="ABC")

Modify the home template for setting the new campaign name
    vim templates/home.html +74
        <h1>Campaign ABC</h1>

    python manage.py shell
        from labelMeNav import models

Set all stages as disabled
    models.Stage.objects.filter().update(enabled=False)

Count the total number of images, and the number of images set to enabled stages
    models.Image.objects.filter().count()    
    models.Image.objects.filter(stage__enabled=True).count()

Create new stage
    models.Stage.objects.create(name="ABC")

Make sure the new images are set as underscore extension (.jpg)
    ls Images/*jpg
    ls Annotations/*xml

Do an "ls | wc" and make sure both images and annotations have the same number of files.
    ls Images/*jpg | wc -l
    ls Annotations/*xml | wc -l

Copy the images to the dir
    umask 002 
    mkdir /var/www/html/LabelMeAnnotationTool/Images/ABC
    cp Images/*jpg /var/www/html/LabelMeAnnotationTool/Images/ABC/

Copy the annotations to the dir
    umask 002 
    mkdir /var/www/html/LabelMeAnnotationTool/Annotations/ABC
    cp Annotations/*xml /var/www/html/LabelMeAnnotationTool/Annotations/ABC/

Make sure the folder, username, and filename tags were set correctly.
    folder: must match the folder name in which the new annotation files will be copied into
    username: tsukeyoka
    filename: should match the image name with lowercase .jpg extension

Start the service
    screen -R 
    python manage.py runserver 0.0.0.0
    Ctrl + a, d

Go to the browser and run update-db
    http://vm041.bridges.psc.edu:8000/update-db

Once again, count the total number of images, and the number of images set to enabled stages. Make sure the number matches the "ls | wc" of what was just added
    ls /var/www/html/LabelMeAnnotationTool/Images/ABC/*jpg | wc -l
    ls /var/www/html/LabelMeAnnotationTool/Images/Annotations/*xml | wc -l
