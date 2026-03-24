"""Tests for memory app views."""
import pytest
from rest_framework.test import APIClient
from memory.models import ClientProfile, MemoryEntry


@pytest.fixture
def api_client():
    return APIClient()


@pytest.mark.django_db
class TestClientProfileViews:
    def test_list_empty(self, api_client):
        response = api_client.get('/api/memory/profiles/')
        assert response.status_code == 200

    def test_create_profile(self, api_client):
        response = api_client.post(
            '/api/memory/profiles/',
            {'client_name': 'Acme Corp', 'industry': 'retail'},
            format='json',
        )
        assert response.status_code == 201
        assert response.json()['client_name'] == 'Acme Corp'

    def test_retrieve_profile(self, api_client):
        profile = ClientProfile.objects.create(client_name='Acme Corp', industry='retail')
        response = api_client.get(f'/api/memory/profiles/{profile.pk}/')
        assert response.status_code == 200
        assert response.json()['client_name'] == 'Acme Corp'

    def test_update_profile(self, api_client):
        profile = ClientProfile.objects.create(client_name='Acme Corp', industry='retail')
        response = api_client.patch(
            f'/api/memory/profiles/{profile.pk}/',
            {'industry': 'tech'},
            format='json',
        )
        assert response.status_code == 200
        assert response.json()['industry'] == 'tech'

    def test_delete_profile(self, api_client):
        profile = ClientProfile.objects.create(client_name='Acme Corp', industry='retail')
        response = api_client.delete(f'/api/memory/profiles/{profile.pk}/')
        assert response.status_code == 204
        assert not ClientProfile.objects.filter(pk=profile.pk).exists()

    def test_duplicate_client_name_fails(self, api_client):
        ClientProfile.objects.create(client_name='Acme Corp')
        response = api_client.post(
            '/api/memory/profiles/',
            {'client_name': 'Acme Corp'},
            format='json',
        )
        assert response.status_code == 400


@pytest.mark.django_db
class TestMemoryEntryViews:
    def test_list_entries(self, api_client):
        response = api_client.get('/api/memory/entries/')
        assert response.status_code == 200

    def test_create_entry(self, api_client):
        response = api_client.post(
            '/api/memory/entries/',
            {
                'title': 'Cloud strategy insight',
                'content': 'Acme is pursuing cloud-first strategy.',
                'entry_type': 'research_insight',
                'client_name': 'Acme Corp',
            },
            format='json',
        )
        assert response.status_code == 201
        assert response.json()['client_name'] == 'Acme Corp'

    def test_retrieve_entry(self, api_client):
        entry = MemoryEntry.objects.create(
            title='Cloud strategy',
            content='Test insight',
            entry_type='research_insight',
            client_name='Acme Corp',
        )
        response = api_client.get(f'/api/memory/entries/{entry.pk}/')
        assert response.status_code == 200
        assert response.json()['title'] == 'Cloud strategy'

    def test_delete_entry(self, api_client):
        entry = MemoryEntry.objects.create(
            title='To delete', content='x', entry_type='research_insight'
        )
        response = api_client.delete(f'/api/memory/entries/{entry.pk}/')
        assert response.status_code == 204


@pytest.mark.django_db
class TestContextQueryView:
    def test_query_returns_context(self, api_client):
        from unittest.mock import patch, MagicMock
        mock_context = MagicMock()
        mock_context.client_profiles = []
        mock_context.sales_plays = []
        mock_context.memory_entries = []
        mock_context.relevance_summary = 'No context found'
        mock_context.to_prompt_context.return_value = ''

        with patch('memory.views.ContextInjector') as MockInjector:
            mock_injector = MagicMock()
            mock_injector.get_context_for_research.return_value = mock_context
            MockInjector.return_value = mock_injector

            response = api_client.post(
                '/api/memory/context/',
                {'client_name': 'Acme Corp'},
                format='json',
            )
        assert response.status_code == 200
        assert 'client_profiles' in response.json()
        assert 'relevance_summary' in response.json()

    def test_query_requires_client_name(self, api_client):
        response = api_client.post('/api/memory/context/', {}, format='json')
        assert response.status_code == 400


@pytest.mark.django_db
class TestCaptureFromResearchView:
    def test_returns_404_for_unknown_job(self, api_client):
        import uuid
        response = api_client.post(f'/api/memory/capture/{uuid.uuid4()}/')
        assert response.status_code == 404

    def test_returns_400_for_non_completed_job(self, api_client):
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Acme Corp', status='running')
        response = api_client.post(f'/api/memory/capture/{job.id}/')
        assert response.status_code == 400

    def test_captures_from_completed_job(self, api_client):
        from unittest.mock import patch, MagicMock
        from research.models import ResearchJob
        job = ResearchJob.objects.create(client_name='Acme Corp', status='completed')

        with patch('memory.views.MemoryCapture') as MockCapture:
            mock_capture = MagicMock()
            mock_capture.capture_from_research.return_value = {'captured': 3}
            MockCapture.return_value = mock_capture

            response = api_client.post(f'/api/memory/capture/{job.id}/')
        assert response.status_code == 200
        assert response.json() == {'captured': 3}
