"""Views for the assets API."""
from django.http import HttpResponse
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from .models import Persona, OnePager, AccountPlan, Citation
from .serializers import (
    PersonaSerializer,
    OnePagerSerializer,
    AccountPlanSerializer,
    CitationSerializer,
    GenerateAssetSerializer,
)
from .services import PersonaGenerator, OnePagerGenerator, AccountPlanGenerator, ExportService
from research.models import ResearchJob
from ideation.models import UseCase


def _get_completed_job(research_job_id):
    """Fetch a completed ResearchJob or return an error Response.

    Returns:
        (ResearchJob, None) on success
        (None, Response) when job not found or not completed
    """
    try:
        job = ResearchJob.objects.get(pk=research_job_id)
    except ResearchJob.DoesNotExist:
        return None, Response({'error': 'Research job not found'}, status=status.HTTP_404_NOT_FOUND)
    if job.status != 'completed':
        return None, Response({'error': 'Research job is not completed'}, status=status.HTTP_400_BAD_REQUEST)
    return job, None


class PersonaListView(generics.ListAPIView):
    """List personas, optionally filtered by research job."""

    serializer_class = PersonaSerializer

    def get_queryset(self):
        queryset = Persona.objects.all()
        research_job_id = self.request.query_params.get('research_job')
        if research_job_id:
            queryset = queryset.filter(research_job_id=research_job_id)
        return queryset


class PersonaDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a persona."""

    queryset = Persona.objects.all()
    serializer_class = PersonaSerializer


class GeneratePersonasView(APIView):
    """Generate buyer personas from research (AGE-21)."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request):
        """Generate personas for a research job.

        Request body:
        {
            "research_job_id": "uuid"
        }
        """
        serializer = GenerateAssetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        job, err = _get_completed_job(serializer.validated_data['research_job_id'])
        if err:
            return err

        # Return existing personas if already generated (avoid wasted Gemini call)
        existing = list(Persona.objects.filter(research_job=job))
        if existing:
            output_serializer = PersonaSerializer(existing, many=True)
            return Response(output_serializer.data, status=status.HTTP_200_OK)

        # Generate personas
        generator = PersonaGenerator()
        persona_data = generator.generate_personas(job)
        personas = generator.create_persona_models(job, persona_data)

        # Serialize and return
        output_serializer = PersonaSerializer(personas, many=True)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)


class OnePagerListView(generics.ListAPIView):
    """List one-pagers, optionally filtered by research job."""

    serializer_class = OnePagerSerializer

    def get_queryset(self):
        queryset = OnePager.objects.all()
        research_job_id = self.request.query_params.get('research_job')
        if research_job_id:
            queryset = queryset.filter(research_job_id=research_job_id)
        return queryset


class OnePagerDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a one-pager."""

    queryset = OnePager.objects.all()
    serializer_class = OnePagerSerializer


class GenerateOnePagerView(APIView):
    """Generate a one-pager from research (AGE-22)."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request):
        """Generate a one-pager for a research job.

        Request body:
        {
            "research_job_id": "uuid",
            "use_case_id": "uuid" (optional)
        }
        """
        serializer = GenerateAssetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        job, err = _get_completed_job(serializer.validated_data['research_job_id'])
        if err:
            return err

        # Return existing one-pager if already generated and no specific use_case requested
        use_case_id = serializer.validated_data.get('use_case_id')
        if not use_case_id:
            existing = OnePager.objects.filter(research_job=job, use_case__isnull=True).first()
            if existing:
                output_serializer = OnePagerSerializer(existing)
                return Response(output_serializer.data, status=status.HTTP_200_OK)

        # Get optional use case
        use_case = None
        use_case_id = serializer.validated_data.get('use_case_id')
        if use_case_id:
            try:
                use_case = UseCase.objects.get(pk=use_case_id)
            except UseCase.DoesNotExist:
                return Response(
                    {'error': 'Use case not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

        # Generate one-pager
        generator = OnePagerGenerator()
        one_pager_data = generator.generate_one_pager(job, use_case)
        one_pager = generator.create_one_pager_model(job, one_pager_data)

        # Generate HTML
        export_service = ExportService()
        export_service.generate_one_pager_html(one_pager)

        # Serialize and return
        output_serializer = OnePagerSerializer(one_pager)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)


class OnePagerHtmlView(APIView):
    """Get one-pager as HTML."""

    def get(self, request, pk):
        try:
            one_pager = OnePager.objects.get(pk=pk)
        except OnePager.DoesNotExist:
            return Response(
                {'error': 'One-pager not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        if not one_pager.html_content:
            export_service = ExportService()
            export_service.generate_one_pager_html(one_pager)

        return HttpResponse(one_pager.html_content, content_type='text/html')


class AccountPlanListView(generics.ListAPIView):
    """List account plans, optionally filtered by research job."""

    serializer_class = AccountPlanSerializer

    def get_queryset(self):
        queryset = AccountPlan.objects.all()
        research_job_id = self.request.query_params.get('research_job')
        if research_job_id:
            queryset = queryset.filter(research_job_id=research_job_id)
        return queryset


class AccountPlanDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete an account plan."""

    queryset = AccountPlan.objects.all()
    serializer_class = AccountPlanSerializer


class GenerateAccountPlanView(APIView):
    """Generate an account plan from research (AGE-23)."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request):
        """Generate an account plan for a research job.

        Request body:
        {
            "research_job_id": "uuid"
        }
        """
        serializer = GenerateAssetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        job, err = _get_completed_job(serializer.validated_data['research_job_id'])
        if err:
            return err

        # Return existing plan if already generated (avoid wasted Gemini call)
        existing = AccountPlan.objects.filter(research_job=job).first()
        if existing:
            output_serializer = AccountPlanSerializer(existing)
            return Response(output_serializer.data, status=status.HTTP_200_OK)

        # Generate account plan
        generator = AccountPlanGenerator()
        plan_data = generator.generate_account_plan(job)
        plan = generator.create_account_plan_model(job, plan_data)

        # Generate HTML
        export_service = ExportService()
        export_service.generate_account_plan_html(plan)

        # Serialize and return
        output_serializer = AccountPlanSerializer(plan)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)


class AccountPlanHtmlView(APIView):
    """Get account plan as HTML."""

    def get(self, request, pk):
        try:
            plan = AccountPlan.objects.get(pk=pk)
        except AccountPlan.DoesNotExist:
            return Response(
                {'error': 'Account plan not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        if not plan.html_content:
            export_service = ExportService()
            export_service.generate_account_plan_html(plan)

        return HttpResponse(plan.html_content, content_type='text/html')


class CitationListView(generics.ListCreateAPIView):
    """List and create citations."""

    serializer_class = CitationSerializer

    def get_queryset(self):
        queryset = Citation.objects.all()
        research_job_id = self.request.query_params.get('research_job')
        if research_job_id:
            queryset = queryset.filter(research_job_id=research_job_id)
        return queryset


class CitationDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a citation."""

    queryset = Citation.objects.all()
    serializer_class = CitationSerializer
