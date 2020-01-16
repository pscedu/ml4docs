from django import template
from django.http import HttpResponse
from django.template import loader
from django.views import View
import labelMeNav.models as models
import labelMeNav.constants as constants


class Home(View):

    def get(self, request):
        template = loader.get_template('home.html')
        context = {
            "color_code": constants.color_code,
            "folder_name": "Birds",
            "images": models.Images.objects.all(),
            # "default_domain": request.build_absolute_uri()
            "default_domain": "http://128.237.138.63/"
        }

        return HttpResponse(template.render(context, request))

    post = get
