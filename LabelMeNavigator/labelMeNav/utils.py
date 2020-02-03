import glob
import shutil

from labelMeNav import constants
from labelMeNav import models


def get_image_names(directory: str = None) -> list:
    files = glob.glob(directory + "/*jpg")
    files = [file.replace(".jpg", "").replace(directory + "/", "") for file in files]
    return files


def get_image_name_list():
    return get_image_names(directory=constants.IMAGES_DIR_PATH)


def machine_labeled_updater():
    """
    Get a zip file with annotations inside for: 1) setting the annotations in the directory they should be, and
    2) changing the status-flags for those images in the DB.
    :return: new statuses
    """

    # Get newly labeled file names
    machine_labeled_stamps = []
    machine_labeled_pages = []
    human_labeled_stamps = []
    human_labeled_pages = []
    files = get_image_names(directory=constants.NEW_ANNOTATIONS_INPUT_FOLDER)

    # Parse the file names and divide them into stamps or pages. Removing prefix as it is not necessary after this.
    for file in files:
        if "".startswith(constants.MACHINE_LABELED_STAMPS_PREFIX):
            file.replace(constants.MACHINE_LABELED_STAMPS_PREFIX, "")
            machine_labeled_stamps.append(file)
        elif "".startswith(constants.MACHINE_LABELED_PAGES_PREFIX):
            file.replace(constants.MACHINE_LABELED_PAGES_PREFIX, "")
            machine_labeled_pages.append(file)
        elif "".startswith(constants.HUMAN_LABELED_STAMPS_PREFIX):
            file.replace(constants.HUMAN_LABELED_STAMPS_PREFIX, "")
            human_labeled_stamps.append(file)
        elif "".startswith(constants.HUMAN_LABELED_PAGES_PREFIX):
            file.replace(constants.HUMAN_LABELED_PAGES_PREFIX, "")
            human_labeled_pages.append(file)

    # Move the files to the Annotations folder. Keep track of mv errors. Object updates will not be performed on those.
    error_list = []
    for file in list(machine_labeled_stamps):
        result = shutil.move(
            src=constants.NEW_ANNOTATIONS_INPUT_FOLDER + '/' + constants.MACHINE_LABELED_STAMPS_PREFIX + '/' + file + ".jpg",
            dst=constants.IMAGES_DIR_PATH + '/')
        if not result:
            error_list.append(file)
            machine_labeled_stamps.remove(file)

    for file in list(machine_labeled_pages):
        result = shutil.move(
            src=constants.NEW_ANNOTATIONS_INPUT_FOLDER + '/' + constants.MACHINE_LABELED_PAGES_PREFIX + '/' + file + ".jpg",
            dst=constants.IMAGES_DIR_PATH + '/')
        if not result:
            error_list.append(file)
            machine_labeled_pages.remove(file)

    for file in list(human_labeled_stamps):
        result = shutil.move(
            src=constants.NEW_ANNOTATIONS_INPUT_FOLDER + '/' + constants.HUMAN_LABELED_STAMPS_PREFIX + '/' + file + ".jpg",
            dst=constants.IMAGES_DIR_PATH + '/')
        if not result:
            error_list.append(file)
            human_labeled_stamps.remove(file)

    for file in list(human_labeled_pages):
        result = shutil.move(
            src=constants.NEW_ANNOTATIONS_INPUT_FOLDER + '/' + constants.HUMAN_LABELED_PAGES_PREFIX + '/' + file + ".jpg",
            dst=constants.IMAGES_DIR_PATH + '/')
        if not result:
            error_list.append(file)
            human_labeled_pages.remove(file)

    models.Image.objects.update_statuses(machine_labeled_stamps=machine_labeled_stamps,
                                         machine_labeled_pages=machine_labeled_pages)


def clean_post_data(data):
    for key, value in data.items():
        if value or value == '':
            if isinstance(value, str):
                value = value.strip()
            if value in ('True', 'true'):
                data[key] = True
            elif value in ('False', 'false'):
                data[key] = False
            elif value in ('', 'None'):
                data[key] = None
    return data
