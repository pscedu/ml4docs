from django.db import models

from labelMeNav import constants


class ImageManager(models.Manager):
    def update_statuses(self, machine_labeled_stamps: list = None, machine_labeled_pages: list = None):
        if machine_labeled_stamps:
            self.filter(image_name__in=machine_labeled_stamps).update(machine_stamps_complete=True)
        elif machine_labeled_pages:
            self.filter(image_name__in=machine_labeled_pages).update(machine_pages_complete=True)

    def labeled_stamps(self):
        return self.filter(human_labeled_stamps=True)

    def pending_stamps(self):
        return self.filter(human_labeled_stamps=False, machine_labeled_stamps=False)

    def labeled_pages(self):
        return self.filter(human_labeled_pages=True)

    def pending_pages(self):
        return self.filter(human_labeled_pages=False, machine_labeled_pages=False)

    def get_next_pending_images(self):
        labeled_stamps = self.labeled_stamps()
        labeled_stamps_count = labeled_stamps.count()

        pending_stamps = self.pending_stamps()
        pending_stamps_count = pending_stamps.count()

        labeled_pages = self.labeled_pages()
        labeled_pages_count = labeled_pages.count()

        pending_pages = self.pending_pages()
        pending_pages_count = pending_pages.count()

        context = {
            "labeled_stamps": labeled_stamps[:10],
            "labeled_stamps_count": labeled_stamps_count,

            "pending_stamps": pending_stamps[:10],
            "pending_stamps_count": pending_stamps_count,

            "labeled_pages": labeled_pages[:10],
            "labeled_pages_count": labeled_pages_count,

            "pending_pages": pending_pages[:10],
            "pending_pages_count": pending_pages_count,
        }
        return context


class Image(models.Model):
    image_name = models.CharField(max_length=200)
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
