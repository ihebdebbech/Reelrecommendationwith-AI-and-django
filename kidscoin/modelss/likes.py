from django.db import models

class like(models.Model):
 
    # Define your Reel model fields here
   
    Userid = models.CharField(max_length=100)
    ReelId = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.Userid

# Additional Reel-related models if needed
    class Meta:
        # Specify the existing collection name explicitly
        db_table = "likes"