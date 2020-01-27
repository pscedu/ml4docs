from django.db import models

from labelMeNav import constants


class ImageManager(models.Manager):
    def update_statuses(self, machine_labeled_stamps: list = None, machine_labeled_pages: list = None):
        if machine_labeled_stamps:
            self.filter(image_file_name__in=machine_labeled_stamps).update(machine_stamps_complete=True)
        elif machine_labeled_pages:
            self.filter(image_file_name__in=machine_labeled_pages).update(machine_pages_complete=True)


class Image(models.Model):
    image_file = models.CharField(max_length=200)
    # End state of the data
    human_labeled_stamps = models.BooleanField(default=False)
    human_labeled_pages = models.BooleanField(default=False)
    machine_labeled_stamps = models.BooleanField(default=False)
    machine_labeled_pages = models.BooleanField(default=False)

    objects = ImageManager()

    @property
    def pending(self):
        return self.human_labeled_stamps and self.human_labeled_pages and \
               self.machine_labeled_stamps and self.machine_labeled_pages

    @property
    def verified(self):
        return not self.pending

    @property
    def status_color_rgb(self):
        if self.human_labeled_stamps:
            return constants.HUMAN_STAMPS_COMPLETE
        elif self.human_labeled_pages:
            return constants.HUMAN_PAGES_COMPLETE
        elif self.machine_labeled_stamps:
            return constants.MACHINE_PAGES_COMPLETE
        elif self.machine_labeled_pages:
            return constants.MACHINE_STAMPS_COMPLETE
        else:
            return constants.PENDING
