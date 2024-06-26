# Generated by Django 4.1.13 on 2024-04-23 17:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('kidscoin', '0007_like_delete_likes'),
    ]

    operations = [
        migrations.CreateModel(
            name='Reels',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('caption', models.TextField()),
                ('hashtags', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'reels',
            },
        ),
        migrations.DeleteModel(
            name='Reel',
        ),
    ]
