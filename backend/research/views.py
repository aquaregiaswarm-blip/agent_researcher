import logging
import threading
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import ResearchJob, ResearchReport, CompetitorCaseStudy, GapAnalysis
from .serializers import (
    ResearchJobCreateSerializer,
    ResearchJobDetailSerializer,
    ResearchReportSerializer,
    CompetitorCaseStudySerializer,
    GapAnalysisSerializer,
)
from .graph import research_workflow

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

    queryset = ResearchJob.objects.all().order_by('-created_at')
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
