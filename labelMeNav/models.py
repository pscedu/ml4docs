from django.db import models

from labelMeNav import constants


class Images(models.Model):
    image_file_name = models.CharField(max_length=200)
    # End state of the data
    human_stamps_complete = models.BooleanField(default=False)
    human_pages_complete = models.BooleanField(default=False)
    machine_stamps_complete = models.BooleanField(default=False)
    machine_pages_complete = models.BooleanField(default=False)

    @property
    def pending(self):
        return self.human_stamps_complete and self.human_pages_complete and \
               self.machine_stamps_complete and self.machine_pages_complete

    @property
    def verified(self):
        return not self.pending

    @property
    def status_color_rgb(self):
        if self.human_stamps_complete:
            return constants.HUMAN_STAMPS_COMPLETE
        elif self.human_pages_complete:
            return constants.HUMAN_PAGES_COMPLETE
        elif self.machine_stamps_complete:
            return constants.MACHINE_PAGES_COMPLETE
        elif self.machine_pages_complete:
            return constants.MACHINE_STAMPS_COMPLETE
        else:
            return constants.PENDING
