"""Management command to recover research jobs stuck in 'running' status."""
import logging
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Mark 'running' jobs as 'completed' when their ResearchReport was saved (finalization succeeded but HTTP timed out)"

    def add_arguments(self, parser):
        parser.add_argument(
            "job_ids",
            nargs="*",
            type=str,
            help="Specific ResearchJob UUIDs to fix (omit to auto-detect all stuck jobs)",
        )

    def handle(self, *args, **options):
        from research.models import ResearchJob, ResearchReport

        job_ids = options["job_ids"]

        if job_ids:
            jobs = ResearchJob.objects.filter(id__in=job_ids)
        else:
            # Auto-detect: running jobs that have a ResearchReport (finalization ran)
            jobs = ResearchJob.objects.filter(
                status="running",
                report__isnull=False,
            )

        if not jobs.exists():
            self.stdout.write(self.style.SUCCESS("No stuck jobs found."))
            return

        self.stdout.write(f"Found {jobs.count()} stuck job(s):")

        for job in jobs:
            self.stdout.write(f"  {job.id} ({job.client_name}) — status={job.status}")
            job.status = "completed"
            job.save(update_fields=["status"])
            self.stdout.write(self.style.SUCCESS(f"    -> set to completed"))
