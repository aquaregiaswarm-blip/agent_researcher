"""Views for the ideation API."""
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from .models import UseCase, FeasibilityAssessment, RefinedPlay
from .serializers import (
    UseCaseSerializer,
    FeasibilityAssessmentSerializer,
    RefinedPlaySerializer,
    UseCaseCreateSerializer,
)
from .services import UseCaseGenerator, FeasibilityService, PlayRefiner
from research.models import ResearchJob


class UseCaseListView(generics.ListAPIView):
    """List use cases, optionally filtered by research job."""

    serializer_class = UseCaseSerializer

    def get_queryset(self):
        queryset = UseCase.objects.all()
        research_job_id = self.request.query_params.get('research_job')
        if research_job_id:
            queryset = queryset.filter(research_job_id=research_job_id)
        return queryset


class UseCaseDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a use case."""

    queryset = UseCase.objects.all()
    serializer_class = UseCaseSerializer


class GenerateUseCasesView(APIView):
    """Generate use cases from a completed research job (AGE-18)."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request):
        """Generate use cases for a research job.

        Request body:
        {
            "research_job_id": "uuid"
        }
        """
        serializer = UseCaseCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            job = ResearchJob.objects.get(pk=serializer.validated_data['research_job_id'])
        except ResearchJob.DoesNotExist:
            return Response(
                {'error': 'Research job not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        if job.status != 'completed':
            return Response(
                {'error': 'Research job is not completed'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Generate use cases
        generator = UseCaseGenerator()
        use_case_data = generator.generate_use_cases(job)
        use_cases = generator.create_use_case_models(job, use_case_data)

        # Serialize and return
        output_serializer = UseCaseSerializer(use_cases, many=True)
        return Response(output_serializer.data, status=status.HTTP_201_CREATED)


class AssessFeasibilityView(APIView):
    """Assess feasibility of a use case (AGE-19)."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request, pk):
        """Assess feasibility for a use case.

        Args:
            pk: UseCase UUID
        """
        try:
            use_case = UseCase.objects.get(pk=pk)
        except UseCase.DoesNotExist:
            return Response(
                {'error': 'Use case not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Assess feasibility
        service = FeasibilityService()
        feasibility_data = service.assess_feasibility(use_case)
        assessment = service.create_assessment_model(use_case, feasibility_data)

        # Serialize and return
        output_serializer = FeasibilityAssessmentSerializer(assessment)
        return Response(output_serializer.data)


class RefinePlayView(APIView):
    """Refine a use case into a sales play (AGE-20)."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ai_execute'

    def post(self, request, pk):
        """Refine a use case into a play.

        Args:
            pk: UseCase UUID
        """
        try:
            use_case = UseCase.objects.get(pk=pk)
        except UseCase.DoesNotExist:
            return Response(
                {'error': 'Use case not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Refine into play
        refiner = PlayRefiner()
        play_data = refiner.refine_play(use_case)
        play = refiner.create_play_model(use_case, play_data)

        # Serialize and return
        output_serializer = RefinedPlaySerializer(play)
        return Response(output_serializer.data)


class RefinedPlayListView(generics.ListAPIView):
    """List refined plays."""

    queryset = RefinedPlay.objects.all()
    serializer_class = RefinedPlaySerializer


class RefinedPlayDetailView(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update, or delete a refined play."""

    queryset = RefinedPlay.objects.all()
    serializer_class = RefinedPlaySerializer
