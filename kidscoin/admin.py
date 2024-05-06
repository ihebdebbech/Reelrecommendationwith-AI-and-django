from django.contrib import admin

# Register your models here.
from kidscoin.modelss.reels import Reels
from kidscoin.modelss.likes import like
admin.site.register(Reels)
admin.site.register(like)