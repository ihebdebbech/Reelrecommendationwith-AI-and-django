from django.urls import path,include
from kidscoin import views

from django.conf.urls.static import static
from django.conf import settings

urlpatterns=[
   path('reel',views.ReelApi),
   path('reel/<str:reel_id>/', views.ReelApi),
   path('reelrecommend', views.recommendreelapi),
   path('updatelike', views.addalike),
]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)