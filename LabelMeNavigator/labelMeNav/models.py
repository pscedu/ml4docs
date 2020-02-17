from django.db import models

from labelMeNav import constants

class Stage(models.Model):
    name = models.CharField(default="", max_length=64)
    enabled = models.BooleanField(default=True)

class ImageManager(models.Manager):
    def update_statuses(self,
                        machine_labeled_stamps: list = None,
                        machine_labeled_pages: list = None,
                        human_labeled_stamps: list = None,
                        human_labeled_pages: list = None):

        for image in machine_labeled_stamps:
            self.update_or_create(image_name=image, human_labeled_stamps=True)

        for image in machine_labeled_pages:
            self.update_or_create(image_name=image, human_labeled_pages=True)

        for image in human_labeled_stamps:
            self.update_or_create(image_name=image, machine_labeled_stamps=True)

        for image in human_labeled_pages:
            self.update_or_create(image_name=image, machine_labeled_pages=True)

    #def filter_queryset(self):
    #    return super().filter_queryset().filter(stage__enabled=True)

    def labeled_stamps(self):
        return self.filter(stage__enabled=True, human_labeled_stamps=True).order_by('image_name')

    def pending_stamps(self):
        return self.filter(stage__enabled=True, human_labeled_stamps=False, machine_labeled_stamps=False).order_by('image_name')

    def labeled_pages(self):
        return self.filter(stage__enabled=True, human_labeled_pages=True).order_by('image_name')

    def pending_pages(self):
        return self.filter(stage__enabled=True, human_labeled_pages=False, machine_labeled_pages=False).order_by('image_name')

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
            "labeled_stamps": labeled_stamps,
            "labeled_stamps_count": labeled_stamps_count,

            "pending_stamps": pending_stamps,
            "pending_stamps_count": pending_stamps_count,

            "labeled_pages": labeled_pages,
            "labeled_pages_count": labeled_pages_count,

            "pending_pages": pending_pages,
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
    stage = models.ForeignKey(Stage, on_delete=models.PROTECT)

    objects = ImageManager()

    @property
    def status_dict(self):
        return {
            "human_labeled_stamps": self.human_labeled_stamps,
            "human_labeled_pages": self.human_labeled_pages,
            "machine_labeled_stamps": self.machine_labeled_stamps,
            "machine_labeled_pages": self.machine_labeled_pages,
        }

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
