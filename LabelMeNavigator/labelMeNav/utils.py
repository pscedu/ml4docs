import glob
import shutil

from django.db import transaction

from labelMeNav import constants
from labelMeNav import models


def get_image_names(directory: str = None) -> list:
    files = glob.glob(directory + "/*jpg")
    files.sort()
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

    status_combinations_tuple = (
        (machine_labeled_stamps, constants.MACHINE_LABELED_STAMPS_PREFIX),
        (machine_labeled_pages, constants.MACHINE_LABELED_PAGES_PREFIX),
        (human_labeled_stamps, constants.HUMAN_LABELED_STAMPS_PREFIX),
        (human_labeled_pages, constants.HUMAN_LABELED_PAGES_PREFIX)
    )

    # Parse the file names and divide them into stamps or pages. Removing prefix as it is not necessary after this.
    for file in files:
        for files_list, status_prefix in status_combinations_tuple:
            if file.startswith(status_prefix):
                filename_no_ext = file.replace(status_prefix, "")
                files_list.append(filename_no_ext)

    # Move the files to the Annotations folder. Keep track of mv errors. Object updates will not be performed on those.
    error_list = []

    try:
        with transaction.atomic():
            models.Image.objects.update_statuses(machine_labeled_stamps=machine_labeled_stamps,
                                                 machine_labeled_pages=machine_labeled_pages,
                                                 human_labeled_stamps=human_labeled_stamps,
                                                 human_labeled_pages=human_labeled_pages)

            try:
                for files_list, status_prefix in status_combinations_tuple:
                    for file in list(files_list):
                        result_image = shutil.move(
                            src=constants.NEW_ANNOTATIONS_INPUT_FOLDER + '/' + status_prefix + file + ".jpg",
                            dst=constants.IMAGES_DIR_PATH + '/' + file + ".jpg")

                        result_annotation = shutil.move(
                            src=constants.NEW_ANNOTATIONS_INPUT_FOLDER + '/' + status_prefix + file + ".xml",
                            dst=constants.ANNOTATIONS_DIR_PATH + '/' + file + ".xml")

                        if not (result_image or result_annotation):
                            error_list.append(file)
                            files_list.remove(file)
            except FileNotFoundError:
                error_list.append(
                    "FileNotFoundError. Check if the destination folders exist and are writable by this process: %s %s" % (
                        constants.IMAGES_DIR_PATH + '/', constants.ANNOTATIONS_DIR_PATH + '/'))
            except Exception as e:
                error_list.append("Generic exception: " % e)
    except Exception as e:
        error_list.append("Generic exception: " % e)

    return_dict = {
        "error_list": error_list,
        "machine_labeled_stamps": machine_labeled_stamps,
        "machine_labeled_pages": machine_labeled_pages,
        "human_labeled_stamps": human_labeled_stamps,
        "human_labeled_pages": human_labeled_pages
    }

    return return_dict


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
