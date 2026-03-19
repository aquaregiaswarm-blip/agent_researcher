"""Management command to re-run gap analysis for specific research jobs."""
import logging
from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Re-run gap analysis for one or more research jobs by ID"

    def add_arguments(self, parser):
        parser.add_argument(
            "job_ids",
            nargs="*",
            type=str,
            help="One or more ResearchJob UUIDs to re-run gap analysis for",
        )
        parser.add_argument(
            "--all-corrupt",
            action="store_true",
            help="Find and re-run all jobs whose gap analysis failed to parse",
        )

    def handle(self, *args, **options):
        from research.models import ResearchJob
        from research.services.gemini import GeminiClient
        from research.services.gap_analysis import GapAnalysisService

        gemini_client = GeminiClient()
        gap_service = GapAnalysisService(gemini_client)

        if options["all_corrupt"]:
            corrupt_qs = ResearchJob.objects.filter(
                gap_analysis__analysis_notes__startswith="Analysis parsing failed"
            )
            job_ids = list(corrupt_qs.values_list("id", flat=True))
            job_ids = [str(j) for j in job_ids]
            if not job_ids:
                self.stdout.write(self.style.SUCCESS("No corrupt gap analysis records found."))
                return
            self.stdout.write(f"Found {len(job_ids)} corrupt record(s): {', '.join(job_ids)}")
        else:
            job_ids = options["job_ids"]
            if not job_ids:
                raise CommandError("Provide one or more job IDs, or use --all-corrupt")

        for job_id in job_ids:
            self.stdout.write(f"Processing job {job_id}...")
            try:
                job = ResearchJob.objects.select_related("report").prefetch_related(
                    "competitor_case_studies"
                ).get(id=job_id)
            except ResearchJob.DoesNotExist:
                self.stderr.write(self.style.ERROR(f"  Job {job_id} not found — skipping"))
                continue

            if not job.report:
                self.stderr.write(self.style.WARNING(f"  Job {job_id} has no ResearchReport — skipping"))
                continue

            report = job.report
            competitor_studies = [
                {
                    "competitor_name": cs.competitor_name,
                    "case_study_title": cs.case_study_title,
                    "summary": cs.summary,
                }
                for cs in job.competitor_case_studies.all()
            ]

            try:
                gap_data = gap_service.analyze_gaps(
                    client_name=job.client_name,
                    vertical=job.vertical or "",
                    company_overview=report.company_overview or "",
                    sales_history=job.sales_history or "",
                    pain_points=report.pain_points or [],
                    opportunities=report.opportunities or [],
                    strategic_goals=report.strategic_goals or [],
                    key_initiatives=report.key_initiatives or [],
                    digital_maturity=report.digital_maturity or "",
                    ai_footprint=report.ai_footprint or "",
                    ai_adoption_stage=report.ai_adoption_stage or "",
                    competitor_case_studies=competitor_studies,
                )

                gap_analysis = gap_service.create_gap_analysis_model(job, gap_data)

                if gap_data.analysis_notes.startswith("Analysis parsing failed"):
                    self.stderr.write(self.style.WARNING(
                        f"  Job {job_id} ({job.client_name}): gap analysis parsing failed again"
                    ))
                else:
                    self.stdout.write(self.style.SUCCESS(
                        f"  Job {job_id} ({job.client_name}): gap analysis updated "
                        f"(confidence={gap_data.confidence_score:.0%}, "
                        f"gaps={len(gap_data.technology_gaps)}T/{len(gap_data.capability_gaps)}C/{len(gap_data.process_gaps)}P)"
                    ))

            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"  Job {job_id} ({job.client_name}): failed — {exc}"))
                logger.exception("rerun_gap_analysis failed for job %s", job_id)
