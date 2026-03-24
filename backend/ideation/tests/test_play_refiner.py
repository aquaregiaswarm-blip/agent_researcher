"""Tests for play_refiner service."""
import json
import pytest
from unittest.mock import MagicMock
from ideation.services.play_refiner import PlayRefiner, RefinedPlayData


def make_mock_gemini(return_value):
    mock = MagicMock()
    mock.generate_text.return_value = return_value
    return mock


def make_use_case():
    uc = MagicMock()
    uc.title = 'AI Forecasting'
    uc.description = 'ML forecasting solution'
    uc.business_problem = 'Excess inventory'
    uc.proposed_solution = 'Azure ML pipeline'
    uc.expected_benefits = ['30% cost reduction']
    uc.estimated_roi = '$1M'
    uc.time_to_value = '6 months'
    uc.research_job.client_name = 'Acme Corp'
    uc.research_job.vertical = 'retail'
    uc.research_job.report.digital_maturity = 'developing'
    uc.feasibility_assessment = None
    return uc


class TestRefinedPlayData:
    def test_default_values(self):
        play = RefinedPlayData()
        assert play.title == ''
        assert play.elevator_pitch == ''
        assert isinstance(play.key_differentiators, list)

    def test_custom_values(self):
        play = RefinedPlayData(title='My Play', elevator_pitch='30s pitch')
        assert play.title == 'My Play'
        assert play.elevator_pitch == '30s pitch'


class TestPlayRefiner:
    def test_refine_play_parses_valid_json(self):
        refiner = PlayRefiner()
        refiner._gemini_client = make_mock_gemini(json.dumps({
            'title': 'AI Forecasting Sales Play',
            'elevator_pitch': 'Cut inventory costs by 30% with AI.',
            'value_proposition': 'Accurate demand forecasting.',
            'key_differentiators': ['Speed', 'Accuracy'],
            'target_persona': 'CIO',
            'target_vertical': 'retail',
            'company_size_fit': 'Mid-market',
            'discovery_questions': ['How do you manage inventory today?'],
            'objection_handlers': [
                {'objection': 'Too expensive', 'response': 'ROI in 6 months'}
            ],
            'proof_points': ['Case study: 30% reduction'],
            'competitive_positioning': 'Best in class',
            'next_steps': ['Schedule demo'],
            'success_metrics': ['Inventory cost reduction'],
        }))

        result = refiner.refine_play(make_use_case())
        assert result.title == 'AI Forecasting Sales Play'
        assert result.elevator_pitch == 'Cut inventory costs by 30% with AI.'
        assert result.key_differentiators == ['Speed', 'Accuracy']
        assert result.target_persona == 'CIO'

    def test_refine_play_handles_string_objection_handlers(self):
        refiner = PlayRefiner()
        refiner._gemini_client = make_mock_gemini(json.dumps({
            'title': 'Test',
            'elevator_pitch': '',
            'value_proposition': '',
            'key_differentiators': [],
            'target_persona': '',
            'target_vertical': '',
            'company_size_fit': '',
            'discovery_questions': [],
            'objection_handlers': ['Too expensive'],  # string format
            'proof_points': [],
            'competitive_positioning': '',
            'next_steps': [],
            'success_metrics': [],
        }))

        result = refiner.refine_play(make_use_case())
        assert len(result.objection_handlers) == 1
        assert result.objection_handlers[0]['objection'] == 'Too expensive'
        assert result.objection_handlers[0]['response'] == ''

    def test_refine_play_strips_markdown_fences(self):
        refiner = PlayRefiner()
        refiner._gemini_client = make_mock_gemini(
            '```json\n{"title": "Test", "elevator_pitch": "x", "value_proposition": "y", '
            '"key_differentiators": [], "target_persona": "", "target_vertical": "", '
            '"company_size_fit": "", "discovery_questions": [], "objection_handlers": [], '
            '"proof_points": [], "competitive_positioning": "", "next_steps": [], '
            '"success_metrics": []}\n```'
        )
        result = refiner.refine_play(make_use_case())
        assert result.title == 'Test'

    def test_refine_play_returns_default_on_invalid_json(self):
        refiner = PlayRefiner()
        refiner._gemini_client = make_mock_gemini('bad json')
        uc = make_use_case()
        result = refiner.refine_play(uc)
        assert isinstance(result, RefinedPlayData)
        assert result.title == uc.title

    def test_refine_play_returns_default_on_exception(self):
        refiner = PlayRefiner()
        mock = MagicMock()
        mock.generate_text.side_effect = Exception('API down')
        refiner._gemini_client = mock
        uc = make_use_case()
        result = refiner.refine_play(uc)
        assert isinstance(result, RefinedPlayData)
        assert result.title == uc.title

    def test_refine_play_uses_feasibility_context_when_available(self):
        refiner = PlayRefiner()
        refiner._gemini_client = make_mock_gemini(json.dumps({
            'title': 'Test', 'elevator_pitch': '', 'value_proposition': '',
            'key_differentiators': [], 'target_persona': '', 'target_vertical': '',
            'company_size_fit': '', 'discovery_questions': [], 'objection_handlers': [],
            'proof_points': [], 'competitive_positioning': '', 'next_steps': [],
            'success_metrics': [],
        }))

        uc = make_use_case()
        assessment = MagicMock()
        assessment.overall_feasibility = 'high'
        assessment.technical_risks = ['Data quality issue']
        assessment.recommendations = 'Proceed with pilot'
        uc.feasibility_assessment = assessment

        result = refiner.refine_play(uc)
        # Verify that the prompt included feasibility context
        call_args = refiner._gemini_client.generate_text.call_args[0][0]
        assert 'high' in call_args
        assert 'Data quality issue' in call_args
