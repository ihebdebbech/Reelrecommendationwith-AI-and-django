from django.db import models
from bson import ObjectId
class Reels(models.Model):
    _id = models.CharField(max_length=50, default=str(ObjectId()))
    # Define your Reel model fields here
    name = models.CharField(max_length=100)
    caption = models.TextField()
    hashtags = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

# Additional Reel-related models if needed
    class Meta:
        # Specify the existing collection name explicitly
        db_table = "reels"