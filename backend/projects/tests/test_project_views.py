"""Tests for projects app views."""
import pytest
from rest_framework.test import APIClient
from projects.models import Project, Iteration, WorkProduct, Annotation


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def project(db):
    return Project.objects.create(
        name='Acme AI Initiative',
        client_name='Acme Corp',
        description='Cloud migration project',
        context_mode='accumulate',
    )


@pytest.mark.django_db
class TestProjectViewSet:
    def test_list_projects(self, api_client):
        response = api_client.get('/api/projects/')
        assert response.status_code == 200

    def test_create_project(self, api_client):
        response = api_client.post(
            '/api/projects/',
            {
                'name': 'New Project',
                'client_name': 'Test Corp',
                'context_mode': 'accumulate',
            },
            format='json',
        )
        assert response.status_code == 201
        assert response.json()['name'] == 'New Project'

    def test_retrieve_project(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/')
        assert response.status_code == 200
        assert response.json()['name'] == 'Acme AI Initiative'

    def test_update_project(self, api_client, project):
        response = api_client.patch(
            f'/api/projects/{project.pk}/',
            {'description': 'Updated description'},
            format='json',
        )
        assert response.status_code == 200
        assert response.json()['description'] == 'Updated description'

    def test_delete_project(self, api_client, project):
        response = api_client.delete(f'/api/projects/{project.pk}/')
        assert response.status_code == 204
        assert not Project.objects.filter(pk=project.pk).exists()

    def test_timeline_action(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/timeline/')
        assert response.status_code == 200

    def test_compare_requires_both_params(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/compare/?a=1')
        assert response.status_code == 400

    def test_compare_returns_404_for_invalid_sequence(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/compare/?a=99&b=100')
        assert response.status_code == 404

    def test_compare_returns_comparison_data(self, api_client, project):
        iter_a = Iteration.objects.create(project=project, sequence=1, status='completed')
        iter_b = Iteration.objects.create(project=project, sequence=2, status='completed')
        response = api_client.get(f'/api/projects/{project.pk}/compare/?a=1&b=2')
        assert response.status_code == 200
        data = response.json()
        assert 'iteration_a' in data
        assert 'iteration_b' in data
        assert 'differences' in data


@pytest.mark.django_db
class TestIterationViews:
    def test_list_iterations_for_project(self, api_client, project):
        Iteration.objects.create(project=project, sequence=1)
        response = api_client.get(f'/api/projects/{project.pk}/iterations/')
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_retrieve_iteration(self, api_client, project):
        Iteration.objects.create(project=project, sequence=1, status='pending')
        response = api_client.get(f'/api/projects/{project.pk}/iterations/1/')
        assert response.status_code == 200

    def test_retrieve_iteration_404(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/iterations/99/')
        assert response.status_code == 404


@pytest.mark.django_db
class TestWorkProductViews:
    def test_list_work_products(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/work-products/')
        assert response.status_code == 200

    def test_create_work_product_with_valid_content_type(self, api_client, project):
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Test Co')
        response = api_client.post(
            f'/api/projects/{project.pk}/work-products/',
            {
                'content_type': 'research.researchjob',
                'object_id': str(job.id),
                'category': 'insight',
                'notes': 'Key research finding',
            },
            format='json',
        )
        assert response.status_code == 201

    def test_create_work_product_invalid_content_type(self, api_client, project):
        import uuid
        response = api_client.post(
            f'/api/projects/{project.pk}/work-products/',
            {
                'content_type': 'nonexistent.model',
                'object_id': str(uuid.uuid4()),
                'category': 'insight',
            },
            format='json',
        )
        assert response.status_code == 400

    def test_delete_work_product(self, api_client, project):
        from django.contrib.contenttypes.models import ContentType
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Test Co')
        ct = ContentType.objects.get_for_model(ResearchJob)
        wp = WorkProduct.objects.create(
            project=project,
            content_type=ct,
            object_id=job.id,
            category='insight',
        )
        response = api_client.delete(f'/api/projects/{project.pk}/work-products/{wp.pk}/')
        assert response.status_code == 204


@pytest.mark.django_db
class TestAnnotationViews:
    def test_list_annotations(self, api_client, project):
        response = api_client.get(f'/api/projects/{project.pk}/annotations/')
        assert response.status_code == 200

    def test_create_annotation(self, api_client, project):
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Test Co')
        response = api_client.post(
            f'/api/projects/{project.pk}/annotations/',
            {
                'content_type': 'research.researchjob',
                'object_id': str(job.id),
                'text': 'Key insight about the client.',
            },
            format='json',
        )
        assert response.status_code == 201
        assert response.json()['text'] == 'Key insight about the client.'

    def test_update_annotation(self, api_client, project):
        from django.contrib.contenttypes.models import ContentType
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Test Co')
        ct = ContentType.objects.get_for_model(ResearchJob)
        annotation = Annotation.objects.create(
            project=project,
            content_type=ct,
            object_id=job.id,
            text='Original note',
        )
        response = api_client.patch(
            f'/api/projects/{project.pk}/annotations/{annotation.pk}/',
            {'text': 'Updated note'},
            format='json',
        )
        assert response.status_code == 200
        assert response.json()['text'] == 'Updated note'

    def test_delete_annotation(self, api_client, project):
        from django.contrib.contenttypes.models import ContentType
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Test Co')
        ct = ContentType.objects.get_for_model(ResearchJob)
        annotation = Annotation.objects.create(
            project=project,
            content_type=ct,
            object_id=job.id,
            text='To delete',
        )
        response = api_client.delete(f'/api/projects/{project.pk}/annotations/{annotation.pk}/')
        assert response.status_code == 204
