from django.contrib import admin

import labelMeNav.models as models


class ImagesAdmin(admin.ModelAdmin):
    fields = ['name', ]


admin.site.register(models.Images, ImagesAdmin)
