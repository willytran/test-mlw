from django.urls import path
from . import views

# a namespace for our app, this will become important in the Templates section
app_name = 'mainapp'

# we call the path function to let Django know what of our Python function should be
# called when a certain URL has been entered.
# The name parameter is optional, but lets us later more conveniently link between pages.
urlpatterns = [
  path('', views.index, name='index'),
  path('vis/', views.vis, name='vis'),
  path('highlight/', views.highlight, name='highlight'),
  path('stats/', views.stats, name='stats'),
]