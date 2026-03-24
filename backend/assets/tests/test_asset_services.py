"""Tests for assets service classes (persona, one_pager, account_plan)."""
import json
import pytest
from unittest.mock import MagicMock
from assets.services.persona import PersonaGenerator, PersonaData
from assets.services.one_pager import OnePagerGenerator, OnePagerData
from assets.services.account_plan import AccountPlanGenerator, AccountPlanData


def make_mock_gemini(return_value):
    mock = MagicMock()
    mock.generate_text.return_value = return_value
    return mock


def make_research_job():
    job = MagicMock()
    job.client_name = 'Acme Corp'
    job.vertical = 'retail'
    report = MagicMock()
    report.company_overview = 'A retail company.'
    report.pain_points = ['High costs', 'Manual processes']
    report.opportunities = ['Automation', 'Cloud migration']
    report.digital_maturity = 'developing'
    report.ai_adoption_stage = 'exploring'
    report.strategic_goals = ['Cost reduction', 'Efficiency']
    report.decision_makers = [{'name': 'Jane Doe', 'title': 'CIO'}]
    report.talking_points = ['We reduce costs by 30%']
    report.tech_partnerships = ['Microsoft Azure', 'SAP']
    report.competitive_landscape = 'AWS and Azure compete.'
    job.report = report
    return job


# ── PersonaData ───────────────────────────────────────────────────────────────


class TestPersonaData:
    def test_default_values(self):
        p = PersonaData()
        assert p.name == ''
        assert p.title == ''
        assert isinstance(p.goals, list)

    def test_custom_values(self):
        p = PersonaData(name='Alex Chen', title='CIO', seniority_level='C-Level')
        assert p.name == 'Alex Chen'
        assert p.title == 'CIO'


# ── PersonaGenerator ──────────────────────────────────────────────────────────


class TestPersonaGenerator:
    def test_generate_personas_parses_valid_json(self):
        gen = PersonaGenerator()
        gen._gemini_client = make_mock_gemini(json.dumps({
            'personas': [
                {
                    'name': 'The Tech CIO',
                    'title': 'Chief Information Officer',
                    'department': 'IT',
                    'seniority_level': 'C-Level',
                    'background': 'Former Google engineer',
                    'goals': ['Cost reduction'],
                    'challenges': ['Legacy systems'],
                    'motivations': ['Efficiency'],
                    'decision_criteria': ['ROI'],
                    'preferred_communication': 'Email',
                    'objections': ['Too expensive'],
                    'content_preferences': ['Case studies'],
                    'key_messages': ['30% cost reduction'],
                }
            ]
        }))

        result = gen.generate_personas(make_research_job())
        assert len(result) == 1
        assert result[0].name == 'The Tech CIO'
        assert result[0].title == 'Chief Information Officer'
        assert result[0].goals == ['Cost reduction']

    def test_generate_personas_returns_empty_on_invalid_json(self):
        gen = PersonaGenerator()
        gen._gemini_client = make_mock_gemini('not json')
        result = gen.generate_personas(make_research_job())
        assert result == []

    def test_generate_personas_returns_empty_on_exception(self):
        gen = PersonaGenerator()
        mock = MagicMock()
        mock.generate_text.side_effect = Exception('API error')
        gen._gemini_client = mock
        result = gen.generate_personas(make_research_job())
        assert result == []

    def test_generate_personas_strips_markdown_fences(self):
        gen = PersonaGenerator()
        gen._gemini_client = make_mock_gemini('```json\n{"personas": []}\n```')
        result = gen.generate_personas(make_research_job())
        assert result == []

    @pytest.mark.django_db
    def test_create_persona_models(self):
        from research.models import ResearchJob
        from assets.models import Persona

        job = ResearchJob.objects.create(client_name='Test Co')
        gen = PersonaGenerator()
        persona_data = PersonaData(
            name='Alex Chen',
            title='CIO',
            department='IT',
            seniority_level='C-Level',
            background='Former Google engineer',
            goals=['Cost reduction'],
            challenges=['Legacy systems'],
            motivations=['Efficiency'],
            decision_criteria=['ROI'],
            preferred_communication='Email',
            objections=['Too expensive'],
            content_preferences=['Case studies'],
            key_messages=['30% cost reduction'],
        )
        created = gen.create_persona_models(job, [persona_data])
        assert len(created) == 1
        assert created[0].name == 'Alex Chen'
        assert Persona.objects.filter(research_job=job).count() == 1


# ── OnePagerData ──────────────────────────────────────────────────────────────


class TestOnePagerData:
    def test_default_values(self):
        op = OnePagerData()
        assert op.title == ''
        assert op.headline == ''
        assert op.executive_summary == ''
        assert isinstance(op.differentiators, list)

    def test_custom_values(self):
        op = OnePagerData(title='Acme Cloud Migration', headline='Modernise in 6 months')
        assert op.title == 'Acme Cloud Migration'
        assert op.headline == 'Modernise in 6 months'


# ── OnePagerGenerator ─────────────────────────────────────────────────────────


class TestOnePagerGenerator:
    def test_generate_one_pager_parses_valid_json(self):
        gen = OnePagerGenerator()
        gen._gemini_client = make_mock_gemini(json.dumps({
            'title': 'Acme Cloud Migration',
            'headline': 'Modernise in 6 months',
            'executive_summary': 'Acme needs cloud to scale.',
            'challenge_section': 'Legacy costs $10M/year.',
            'solution_section': 'Azure migration.',
            'benefits_section': '30% cost reduction.',
            'differentiators': ['Speed', 'Cost'],
            'call_to_action': 'Schedule a demo',
            'next_steps': ['Pilot in Q1'],
        }))

        result = gen.generate_one_pager(make_research_job())
        assert result.title == 'Acme Cloud Migration'
        assert result.headline == 'Modernise in 6 months'
        assert result.executive_summary == 'Acme needs cloud to scale.'
        assert result.differentiators == ['Speed', 'Cost']

    def test_generate_one_pager_returns_default_on_invalid_json(self):
        gen = OnePagerGenerator()
        gen._gemini_client = make_mock_gemini('not json')
        result = gen.generate_one_pager(make_research_job())
        assert isinstance(result, OnePagerData)

    def test_generate_one_pager_returns_default_on_exception(self):
        gen = OnePagerGenerator()
        mock = MagicMock()
        mock.generate_text.side_effect = Exception('API down')
        gen._gemini_client = mock
        result = gen.generate_one_pager(make_research_job())
        assert isinstance(result, OnePagerData)

    def test_generate_one_pager_strips_markdown_fences(self):
        gen = OnePagerGenerator()
        gen._gemini_client = make_mock_gemini(
            '```json\n{"title": "Test", "headline": "x", "executive_summary": "",'
            '"challenge_section": "", "solution_section": "", "benefits_section": "",'
            '"html_content": ""}\n```'
        )
        result = gen.generate_one_pager(make_research_job())
        assert result.title == 'Test'


# ── AccountPlanData ───────────────────────────────────────────────────────────


class TestAccountPlanData:
    def test_default_values(self):
        ap = AccountPlanData()
        assert ap.title == ''
        assert ap.executive_summary == ''
        assert isinstance(ap.strategic_objectives, list)

    def test_custom_values(self):
        ap = AccountPlanData(title='Acme Plan', executive_summary='Strategic target')
        assert ap.title == 'Acme Plan'
        assert ap.executive_summary == 'Strategic target'


# ── AccountPlanGenerator ──────────────────────────────────────────────────────


class TestAccountPlanGenerator:
    def test_generate_account_plan_parses_valid_json(self):
        gen = AccountPlanGenerator()
        gen._gemini_client = make_mock_gemini(json.dumps({
            'title': 'Acme Strategic Account Plan',
            'executive_summary': 'Prime target for cloud expansion.',
            'account_overview': 'Global distribution company.',
            'strategic_objectives': ['Cloud-first by 2026'],
            'key_stakeholders': [],
            'opportunities': [],
            'competitive_landscape': 'AWS and Azure compete.',
            'swot_analysis': {
                'strengths': ['Strong brand'],
                'weaknesses': ['Legacy ERP'],
                'opportunities': ['Cloud migration'],
                'threats': ['Incumbent vendor'],
            },
            'engagement_strategy': 'Land and expand via CIO.',
            'value_propositions': ['Reduce TCO by 30%'],
            'action_plan': [],
            'success_metrics': ['50% workloads migrated by Q4'],
            'milestones': [],
            'timeline': {},
        }))

        result = gen.generate_account_plan(make_research_job())
        assert result.title == 'Acme Strategic Account Plan'
        assert result.strategic_objectives == ['Cloud-first by 2026']
        assert result.swot_analysis['strengths'] == ['Strong brand']

    def test_generate_account_plan_returns_default_on_invalid_json(self):
        gen = AccountPlanGenerator()
        gen._gemini_client = make_mock_gemini('not json')
        result = gen.generate_account_plan(make_research_job())
        assert isinstance(result, AccountPlanData)

    def test_generate_account_plan_returns_default_on_exception(self):
        gen = AccountPlanGenerator()
        mock = MagicMock()
        mock.generate_text.side_effect = Exception('API down')
        gen._gemini_client = mock
        result = gen.generate_account_plan(make_research_job())
        assert isinstance(result, AccountPlanData)

    def test_generate_account_plan_strips_markdown_fences(self):
        gen = AccountPlanGenerator()
        plan_json = {
            'title': 'Test Plan', 'executive_summary': '', 'account_overview': '',
            'strategic_objectives': [], 'key_stakeholders': [], 'opportunities': [],
            'competitive_landscape': '',
            'swot_analysis': {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': []},
            'engagement_strategy': '', 'value_propositions': [],
            'action_plan': [], 'success_metrics': [], 'milestones': [], 'timeline': {},
        }
        gen._gemini_client = make_mock_gemini(f'```json\n{json.dumps(plan_json)}\n```')
        result = gen.generate_account_plan(make_research_job())
        assert result.title == 'Test Plan'
