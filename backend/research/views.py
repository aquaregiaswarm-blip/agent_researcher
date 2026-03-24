import logging
import os
import threading
from django.http import FileResponse
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from .models import ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis, InternalOpsIntel
from .serializers import (
    ResearchJobCreateSerializer,
    ResearchJobDetailSerializer,
    ResearchReportSerializer,
    CompetitorCaseStudySerializer,
    GapAnalysisSerializer,
)
from .graph import research_workflow
from assets.services.export import ExportService

logger = logging.getLogger(__name__)


def run_research_sync(job_id: str):
    """Run research workflow synchronously. Called from a dedicated endpoint."""
    job = ResearchJob.objects.get(id=job_id)
    job.status = 'running'
    job.save()

    try:
        initial_state = {
            'client_name': job.client_name,
            'sales_history': job.sales_history,
            'prompt': job.prompt,
            'job_id': str(job.id),
            'status': 'pending',
            'result': '',
            'error': '',
            'research_report': None,
            'vertical': None,
            'competitor_case_studies': None,
            'gap_analysis': None,
            'web_sources': None,
        }

        result = research_workflow.invoke(initial_state)

        job.refresh_from_db()
        job.status = 'completed'
        job.result = result.get('result', '')
        job.error = result.get('error', '')
        if result.get('vertical'):
            job.vertical = result['vertical']
        job.save()
        logger.info(f"Research job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Research job {job_id} failed")
        job.refresh_from_db()
        job.status = 'failed'
        job.error = str(e)
        job.save()


# Keep backward compat for iteration starts (projects/views.py imports this)
def run_research_async(job_id: str):
    """Run research in a thread (legacy). Prefer run_research_sync."""
    run_research_sync(job_id)


class ResearchJobListView(generics.ListAPIView):
    """View for listing all research jobs."""

    queryset = ResearchJob.objects.select_related(
        'report', 'gap_analysis', 'internal_ops'
    ).prefetch_related(
        'competitor_case_studies'
    ).order_by('-created_at')
    serializer_class = ResearchJobDetailSerializer


class ResearchJobCreateView(generics.CreateAPIView):
    """View for creating a new research job."""

    serializer_class = ResearchJobCreateSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()

        # Return job immediately with pending status
        output_serializer = ResearchJobDetailSerializer(job)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)


class ResearchJobExecuteView(APIView):
    """Execute a pending research job synchronously.

    The frontend creates a job (POST /api/research/) then calls this
    endpoint (POST /api/research/<id>/execute/) in the background.
    This runs synchronously within the Cloud Run request timeout (300s),
    avoiding thread-based execution that gets killed on OOM or scale-down.
    """

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request, pk):
        try:
            job = ResearchJob.objects.get(pk=pk)
        except ResearchJob.DoesNotExist:
            return Response(
                {'error': 'Research job not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        if job.status not in ('pending', 'failed'):
            return Response(
                {'error': f'Job is already {job.status}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Run synchronously — Cloud Run timeout is 300s
        run_research_sync(str(job.id))

        job.refresh_from_db()
        serializer = ResearchJobDetailSerializer(job)
        return Response(serializer.data)


class ResearchJobDetailView(generics.RetrieveAPIView):
    """View for retrieving research job details."""

    queryset = ResearchJob.objects.all()
    serializer_class = ResearchJobDetailSerializer


class ResearchReportView(APIView):
    """View for retrieving the structured research report (AGE-10)."""

    def get(self, request, pk):
        try:
            job = ResearchJob.objects.get(pk=pk)
            report = ResearchReport.objects.get(research_job=job)
            serializer = ResearchReportSerializer(report)
            return Response(serializer.data)
        except ResearchJob.DoesNotExist:
            return Response(
                {'error': 'Research job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except ResearchReport.DoesNotExist:
            return Response(
                {'error': 'Research report not yet available'},
                status=status.HTTP_404_NOT_FOUND
            )


class CompetitorCaseStudiesView(APIView):
    """View for retrieving competitor case studies (AGE-12)."""

    def get(self, request, pk):
        try:
            job = ResearchJob.objects.get(pk=pk)
            case_studies = CompetitorCaseStudy.objects.filter(research_job=job)
            serializer = CompetitorCaseStudySerializer(case_studies, many=True)
            return Response(serializer.data)
        except ResearchJob.DoesNotExist:
            return Response(
                {'error': 'Research job not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class GapAnalysisView(APIView):
    """View for retrieving gap analysis (AGE-13)."""

    def get(self, request, pk):
        try:
            job = ResearchJob.objects.get(pk=pk)
            gap_analysis = GapAnalysis.objects.get(research_job=job)
            serializer = GapAnalysisSerializer(gap_analysis)
            return Response(serializer.data)
        except ResearchJob.DoesNotExist:
            return Response(
                {'error': 'Research job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except GapAnalysis.DoesNotExist:
            return Response(
                {'error': 'Gap analysis not yet available'},
                status=status.HTTP_404_NOT_FOUND
            )


class ResearchPdfExportView(APIView):
    """Export complete research as PDF."""

    def get(self, request, pk):
        try:
            # Get job with all related data for efficient queries
            job = ResearchJob.objects.select_related(
                'report',
                'gap_analysis',
                'internal_ops',
            ).prefetch_related(
                'competitor_case_studies',
            ).get(pk=pk)
        except ResearchJob.DoesNotExist:
            return Response(
                {'error': 'Research job not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Validate job is completed
        if job.status != 'completed':
            return Response(
                {'error': f'Cannot export research with status: {job.status}. Research must be completed.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate HTML and PDF
        export_service = ExportService()
        html_content = export_service.generate_research_report_html(job)

        # Generate unique filename
        safe_name = "".join(c if c.isalnum() else "_" for c in job.client_name)
        filename = f"research_{safe_name}_{str(job.id)[:8]}.pdf"

        pdf_path = export_service.export_to_pdf(html_content, filename)

        if not pdf_path:
            return Response(
                {'error': 'Failed to generate PDF. Please ensure weasyprint is installed.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Return file as downloadable response — open handle managed by FileResponse
        file_handle = open(pdf_path, 'rb')
        response = FileResponse(file_handle, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
