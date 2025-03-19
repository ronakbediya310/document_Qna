from django.db import models

# Create your models here.
 

class SourceDocument(models.Model):
    query = models.CharField(max_length=255, unique=True)
    file_path = models.FileField(upload_to="retrieved_docs/", null=True, blank=True)
    content = models.TextField()

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

   

