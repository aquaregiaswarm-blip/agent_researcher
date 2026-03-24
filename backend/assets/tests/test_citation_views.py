"""Tests for assets Citation views (AGE-24)."""
import pytest
from rest_framework.test import APIClient
from assets.models import Citation
from research.models import ResearchJob


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def research_job(db):
    return ResearchJob.objects.create(client_name='Acme Corp', status='completed')


@pytest.fixture
def citation(db, research_job):
    return Citation.objects.create(
        research_job=research_job,
        citation_type='news',
        title='Acme Announces Cloud Migration',
        source='Tech News Daily',
        url='https://technews.example.com/acme-cloud',
        author='Jane Reporter',
        excerpt='Acme Corp announced a major cloud migration initiative.',
        relevance_note='Confirms cloud investment priority',
    )


@pytest.mark.django_db
class TestCitationListView:
    def test_list_returns_200(self, api_client, research_job):
        response = api_client.get(f'/api/assets/citations/?research_job={research_job.id}')
        assert response.status_code == 200

    def test_list_empty_when_no_citations(self, api_client, research_job):
        response = api_client.get(f'/api/assets/citations/?research_job={research_job.id}')
        assert response.json() == []

    def test_list_returns_citations_for_job(self, api_client, research_job, citation):
        response = api_client.get(f'/api/assets/citations/?research_job={research_job.id}')
        data = response.json()
        assert len(data) == 1
        assert data[0]['title'] == 'Acme Announces Cloud Migration'
        assert data[0]['citation_type'] == 'news'

    def test_list_does_not_return_other_jobs_citations(self, api_client, db, citation):
        other_job = ResearchJob.objects.create(client_name='Other Corp')
        response = api_client.get(f'/api/assets/citations/?research_job={other_job.id}')
        assert response.json() == []

    def test_create_citation(self, api_client, research_job):
        response = api_client.post(
            '/api/assets/citations/',
            {
                'research_job': str(research_job.id),
                'citation_type': 'website',
                'title': 'Acme Corp Homepage',
                'source': 'acme.example.com',
                'url': 'https://acme.example.com',
            },
            format='json',
        )
        assert response.status_code == 201
        assert response.json()['title'] == 'Acme Corp Homepage'
        assert response.json()['citation_type'] == 'website'


@pytest.mark.django_db
class TestCitationDetailView:
    def test_retrieve_citation(self, api_client, citation):
        response = api_client.get(f'/api/assets/citations/{citation.pk}/')
        assert response.status_code == 200
        assert response.json()['title'] == 'Acme Announces Cloud Migration'
        assert response.json()['source'] == 'Tech News Daily'

    def test_retrieve_returns_404_for_unknown_id(self, api_client, db):
        import uuid
        response = api_client.get(f'/api/assets/citations/{uuid.uuid4()}/')
        assert response.status_code == 404

    def test_verify_citation_via_patch(self, api_client, citation):
        assert not citation.verified
        response = api_client.patch(
            f'/api/assets/citations/{citation.pk}/',
            {'verified': True},
            format='json',
        )
        assert response.status_code == 200
        assert response.json()['verified'] is True
        citation.refresh_from_db()
        assert citation.verified

    def test_update_citation_fields(self, api_client, citation):
        response = api_client.patch(
            f'/api/assets/citations/{citation.pk}/',
            {'relevance_note': 'Updated: key strategic signal'},
            format='json',
        )
        assert response.status_code == 200
        assert response.json()['relevance_note'] == 'Updated: key strategic signal'

    def test_delete_citation(self, api_client, citation):
        response = api_client.delete(f'/api/assets/citations/{citation.pk}/')
        assert response.status_code == 204
        assert not Citation.objects.filter(pk=citation.pk).exists()
