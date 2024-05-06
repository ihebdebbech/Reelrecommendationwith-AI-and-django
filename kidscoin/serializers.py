from rest_framework import serializers
from .modelss.reels import Reels

class ReelSerializer(serializers.ModelSerializer):
     # Include '_id' field

    class Meta:
        model = Reels
        fields = ['_id','name', 'caption', 'hashtags' ,'created_at']