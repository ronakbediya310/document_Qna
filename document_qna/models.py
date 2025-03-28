from django.db import models
from django.utils.timezone import now
import uuid

# Create your models here.
 

class SourceDocument(models.Model):
    id = models.AutoField(primary_key=True)  
    query = models.CharField(max_length=255,default="sample")
    file_path = models.CharField(max_length=500, null=True, blank=True)
    content = models.TextField()
    source = models.CharField(max_length=255,null=True)
    title = models.CharField(max_length=255,null=True)
    class Meta:
        db_table = "document_qna_sourcedocument"

    def __str__(self):
        return self.query


class UnansweredQuestion(models.Model):
    id = models.AutoField(primary_key=True)  # Ensure primary key is defined
    question = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "document_qna_unansweredquestion" 
class AdminUser(models.Model):
    """Model for Admin Users."""
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=150, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username

class APIToken(models.Model):
    """Model for API Tokens."""
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(AdminUser, on_delete=models.CASCADE)
    token = models.CharField(max_length=255, unique=True, default=uuid.uuid4)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    def is_valid(self):
        """Check if the token is still valid."""
        return self.expires_at > now()

    def __str__(self):
        return f"Token for {self.user.username}"
   

