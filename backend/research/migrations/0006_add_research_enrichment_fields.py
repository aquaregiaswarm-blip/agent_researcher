from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('research', '0005_add_web_sources_to_research_report'),
    ]

    operations = [
        migrations.AddField(
            model_name='researchreport',
            name='cloud_footprint',
            field=models.TextField(blank=True, default=''),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='researchreport',
            name='security_posture',
            field=models.TextField(blank=True, default=''),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='researchreport',
            name='data_maturity',
            field=models.TextField(blank=True, default=''),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='researchreport',
            name='financial_signals',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='researchreport',
            name='tech_partnerships',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
