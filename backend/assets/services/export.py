"""Export service for generating PDFs and HTML (AGE-24)."""
import html
import logging
import os
from datetime import datetime
from typing import Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting assets to HTML and PDF."""

    ONE_PAGER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; color: #333; }}
        .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #0066cc; padding-bottom: 20px; }}
        .headline {{ font-size: 24px; color: #0066cc; margin-bottom: 10px; }}
        .summary {{ font-size: 14px; color: #666; line-height: 1.6; }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{ font-size: 16px; font-weight: bold; color: #0066cc; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .section-content {{ font-size: 14px; line-height: 1.6; }}
        .differentiators {{ list-style: none; padding: 0; }}
        .differentiators li {{ padding: 8px 0 8px 25px; position: relative; }}
        .differentiators li:before {{ content: "✓"; position: absolute; left: 0; color: #0066cc; font-weight: bold; }}
        .cta {{ background: #0066cc; color: white; padding: 15px 25px; border-radius: 5px; text-align: center; margin-top: 30px; }}
        .next-steps {{ margin-top: 15px; }}
        .next-steps li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="headline">{headline}</div>
    </div>

    <div class="section">
        <div class="summary">{executive_summary}</div>
    </div>

    <div class="section">
        <div class="section-title">The Challenge</div>
        <div class="section-content">{challenge_section}</div>
    </div>

    <div class="section">
        <div class="section-title">Our Solution</div>
        <div class="section-content">{solution_section}</div>
    </div>

    <div class="section">
        <div class="section-title">Key Benefits</div>
        <div class="section-content">{benefits_section}</div>
    </div>

    <div class="section">
        <div class="section-title">Why Choose Us</div>
        <ul class="differentiators">
            {differentiators_html}
        </ul>
    </div>

    <div class="cta">{call_to_action}</div>

    <div class="section next-steps">
        <div class="section-title">Next Steps</div>
        <ol>
            {next_steps_html}
        </ol>
    </div>
</body>
</html>
'''

    ACCOUNT_PLAN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; color: #333; }}
        .header {{ text-align: center; margin-bottom: 30px; border-bottom: 3px solid #0066cc; padding-bottom: 20px; }}
        h1 {{ color: #0066cc; }}
        .executive-summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; page-break-inside: avoid; }}
        .section-title {{ font-size: 18px; font-weight: bold; color: #0066cc; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .section-content {{ font-size: 14px; line-height: 1.6; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 13px; }}
        th {{ background: #0066cc; color: white; }}
        .swot {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        .swot-box {{ padding: 15px; border-radius: 5px; }}
        .strengths {{ background: #e8f5e9; }}
        .weaknesses {{ background: #ffebee; }}
        .opportunities {{ background: #e3f2fd; }}
        .threats {{ background: #fff3e0; }}
        .swot-title {{ font-weight: bold; margin-bottom: 10px; }}
        ul {{ margin: 0; padding-left: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
    </div>

    <div class="executive-summary">
        <h2>Executive Summary</h2>
        <p>{executive_summary}</p>
    </div>

    <div class="section">
        <div class="section-title">Account Overview</div>
        <div class="section-content">{account_overview}</div>
    </div>

    <div class="section">
        <div class="section-title">Strategic Objectives</div>
        <ul>
            {strategic_objectives_html}
        </ul>
    </div>

    <div class="section">
        <div class="section-title">Key Stakeholders</div>
        {stakeholders_table}
    </div>

    <div class="section">
        <div class="section-title">Opportunities</div>
        {opportunities_table}
    </div>

    <div class="section">
        <div class="section-title">SWOT Analysis</div>
        <div class="swot">
            <div class="swot-box strengths">
                <div class="swot-title">Strengths</div>
                <ul>{swot_strengths}</ul>
            </div>
            <div class="swot-box weaknesses">
                <div class="swot-title">Weaknesses</div>
                <ul>{swot_weaknesses}</ul>
            </div>
            <div class="swot-box opportunities">
                <div class="swot-title">Opportunities</div>
                <ul>{swot_opportunities}</ul>
            </div>
            <div class="swot-box threats">
                <div class="swot-title">Threats</div>
                <ul>{swot_threats}</ul>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Engagement Strategy</div>
        <div class="section-content">{engagement_strategy}</div>
    </div>

    <div class="section">
        <div class="section-title">Action Plan</div>
        {action_plan_table}
    </div>

    <div class="section">
        <div class="section-title">Success Metrics</div>
        <ul>
            {success_metrics_html}
        </ul>
    </div>

    <div class="section">
        <div class="section-title">Timeline</div>
        <div class="section-content">{timeline}</div>
    </div>
</body>
</html>
'''

    def generate_one_pager_html(self, one_pager) -> str:
        """Generate HTML for a one-pager.

        Args:
            one_pager: OnePager model instance

        Returns:
            HTML string
        """
        differentiators_html = '\n'.join([f'<li>{d}</li>' for d in (one_pager.differentiators or [])])
        next_steps_html = '\n'.join([f'<li>{s}</li>' for s in (one_pager.next_steps or [])])

        html = self.ONE_PAGER_TEMPLATE.format(
            title=one_pager.title,
            headline=one_pager.headline,
            executive_summary=one_pager.executive_summary,
            challenge_section=one_pager.challenge_section,
            solution_section=one_pager.solution_section,
            benefits_section=one_pager.benefits_section,
            differentiators_html=differentiators_html,
            call_to_action=one_pager.call_to_action,
            next_steps_html=next_steps_html,
        )

        # Save HTML to model
        one_pager.html_content = html
        one_pager.save(update_fields=['html_content'])

        return html

    def generate_account_plan_html(self, account_plan) -> str:
        """Generate HTML for an account plan.

        Args:
            account_plan: AccountPlan model instance

        Returns:
            HTML string
        """
        # Strategic objectives
        strategic_objectives_html = '\n'.join([f'<li>{o}</li>' for o in (account_plan.strategic_objectives or [])])

        # Stakeholders table
        stakeholders = account_plan.key_stakeholders or []
        stakeholders_rows = '\n'.join([
            f"<tr><td>{s.get('name', '')}</td><td>{s.get('title', '')}</td><td>{s.get('role_in_decision', '')}</td><td>{s.get('engagement_approach', '')}</td></tr>"
            for s in stakeholders
        ])
        stakeholders_table = f'''
        <table>
            <tr><th>Name</th><th>Title</th><th>Role</th><th>Engagement Approach</th></tr>
            {stakeholders_rows}
        </table>
        ''' if stakeholders_rows else '<p>No stakeholders identified</p>'

        # Opportunities table
        opportunities = account_plan.opportunities or []
        opportunities_rows = '\n'.join([
            f"<tr><td>{o.get('name', '')}</td><td>{o.get('value', '')}</td><td>{o.get('timeline', '')}</td><td>{o.get('probability', '')}</td></tr>"
            for o in opportunities
        ])
        opportunities_table = f'''
        <table>
            <tr><th>Opportunity</th><th>Value</th><th>Timeline</th><th>Probability</th></tr>
            {opportunities_rows}
        </table>
        ''' if opportunities_rows else '<p>No opportunities identified</p>'

        # SWOT
        swot = account_plan.swot_analysis or {}
        swot_strengths = '\n'.join([f'<li>{s}</li>' for s in swot.get('strengths', [])])
        swot_weaknesses = '\n'.join([f'<li>{w}</li>' for w in swot.get('weaknesses', [])])
        swot_opportunities = '\n'.join([f'<li>{o}</li>' for o in swot.get('opportunities', [])])
        swot_threats = '\n'.join([f'<li>{t}</li>' for t in swot.get('threats', [])])

        # Action plan table
        actions = account_plan.action_plan or []
        action_rows = '\n'.join([
            f"<tr><td>{a.get('action', '')}</td><td>{a.get('owner', '')}</td><td>{a.get('due_date', '')}</td><td>{a.get('status', '')}</td></tr>"
            for a in actions
        ])
        action_plan_table = f'''
        <table>
            <tr><th>Action</th><th>Owner</th><th>Due Date</th><th>Status</th></tr>
            {action_rows}
        </table>
        ''' if action_rows else '<p>No actions defined</p>'

        # Success metrics
        success_metrics_html = '\n'.join([f'<li>{m}</li>' for m in (account_plan.success_metrics or [])])

        html = self.ACCOUNT_PLAN_TEMPLATE.format(
            title=account_plan.title,
            executive_summary=account_plan.executive_summary,
            account_overview=account_plan.account_overview,
            strategic_objectives_html=strategic_objectives_html,
            stakeholders_table=stakeholders_table,
            opportunities_table=opportunities_table,
            swot_strengths=swot_strengths,
            swot_weaknesses=swot_weaknesses,
            swot_opportunities=swot_opportunities,
            swot_threats=swot_threats,
            engagement_strategy=account_plan.engagement_strategy,
            action_plan_table=action_plan_table,
            success_metrics_html=success_metrics_html,
            timeline=account_plan.timeline,
        )

        # Save HTML to model
        account_plan.html_content = html
        account_plan.save(update_fields=['html_content'])

        return html

    def export_to_pdf(self, html_content: str, output_filename: str) -> Optional[str]:
        """Export HTML to PDF using weasyprint.

        Args:
            html_content: HTML string to convert
            output_filename: Name of the output file

        Returns:
            Path to the generated PDF, or None if failed
        """
        try:
            from weasyprint import HTML

            # Ensure export directory exists
            export_dir = os.path.join(settings.BASE_DIR, 'exports')
            os.makedirs(export_dir, exist_ok=True)

            output_path = os.path.join(export_dir, output_filename)
            HTML(string=html_content).write_pdf(output_path)

            logger.info(f"Generated PDF: {output_path}")
            return output_path

        except ImportError:
            logger.warning("weasyprint not installed, skipping PDF export")
            return None
        except Exception as e:
            logger.exception(f"Failed to generate PDF: {e}")
            return None

    # ========== Research Report Export ==========

    RESEARCH_REPORT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            font-size: 12px;
            line-height: 1.5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #0066cc;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        .header .vertical-badge {{
            display: inline-block;
            background: #e3f2fd;
            color: #0066cc;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 11px;
            text-transform: capitalize;
            margin-bottom: 10px;
        }}
        .header .generated-date {{
            color: #666;
            font-size: 11px;
        }}
        .page-break {{
            page-break-before: always;
        }}
        .section {{
            margin-bottom: 25px;
            page-break-inside: avoid;
        }}
        .section-title {{
            font-size: 16px;
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 12px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .subsection-title {{
            font-size: 13px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            margin-top: 15px;
        }}
        .stats-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .stat-card {{
            background: #f5f5f5;
            padding: 10px 15px;
            border-radius: 5px;
            text-align: center;
            flex: 1;
            min-width: 100px;
        }}
        .stat-card .label {{
            font-size: 10px;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-weight: bold;
            color: #333;
            font-size: 14px;
        }}
        .card {{
            background: #f9f9f9;
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .card-title {{
            font-weight: bold;
            color: #333;
        }}
        .card-subtitle {{
            font-size: 11px;
            color: #666;
        }}
        .card-content {{
            margin-top: 5px;
            color: #444;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 11px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background: #0066cc;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        ul {{
            margin: 0;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
        .highlight-box {{
            background: #e3f2fd;
            border-left: 4px solid #0066cc;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .warning-box {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .success-box {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        .gap-grid {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .gap-column {{
            flex: 1;
            min-width: 200px;
        }}
        .gap-item {{
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 11px;
        }}
        .gap-tech {{
            background: #ffebee;
            border-left: 3px solid #f44336;
        }}
        .gap-capability {{
            background: #fff3e0;
            border-left: 3px solid #ff9800;
        }}
        .gap-process {{
            background: #f3e5f5;
            border-left: 3px solid #9c27b0;
        }}
        .confidence-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: bold;
        }}
        .confidence-high {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .confidence-medium {{
            background: #fff3e0;
            color: #f57c00;
        }}
        .confidence-low {{
            background: #ffebee;
            color: #c62828;
        }}
        .source-item {{
            padding: 8px 12px;
            background: #f5f5f5;
            border-radius: 5px;
            margin-bottom: 8px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}
        .source-number {{
            background: #0066cc;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
            flex-shrink: 0;
        }}
        .source-content {{
            flex: 1;
        }}
        .source-title {{
            font-weight: bold;
            color: #333;
        }}
        .source-url {{
            font-size: 10px;
            color: #0066cc;
            word-break: break-all;
        }}
        .raw-output {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 10px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: none;
            overflow: visible;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 10px;
            color: #666;
        }}
        .sentiment-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            text-transform: capitalize;
        }}
        .sentiment-positive {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .sentiment-negative {{
            background: #ffebee;
            color: #c62828;
        }}
        .sentiment-neutral {{
            background: #f5f5f5;
            color: #666;
        }}
        .sentiment-mixed {{
            background: #fff3e0;
            color: #f57c00;
        }}
        .rating-bar {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .rating-bar-fill {{
            height: 8px;
            background: #ddd;
            border-radius: 4px;
            flex: 1;
            overflow: hidden;
        }}
        .rating-bar-value {{
            height: 100%;
            background: #0066cc;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    {content}

    <div class="footer">
        Generated by Agent Researcher | {generated_date}
    </div>
</body>
</html>
'''

    def generate_research_report_html(self, research_job) -> str:
        """Generate HTML for a complete research report.

        Args:
            research_job: ResearchJob model instance with related data

        Returns:
            HTML string
        """
        # Get related objects with graceful None handling
        report = getattr(research_job, 'report', None)
        gap_analysis = getattr(research_job, 'gap_analysis', None)
        internal_ops = getattr(research_job, 'internal_ops', None)

        # Handle both prefetched and non-prefetched case studies
        case_studies = list(research_job.competitor_case_studies.all()) if hasattr(research_job, 'competitor_case_studies') else []

        # Build content sections
        sections = []

        # Header
        header_html = self._build_header(research_job)
        sections.append(header_html)

        # Overview section
        overview_html = self._build_overview_section(research_job, report)
        sections.append(overview_html)

        # Deep Research section (page break)
        if report:
            deep_research_html = self._build_deep_research_section(report)
            sections.append(f'<div class="page-break"></div>{deep_research_html}')

        # Competitors section (page break)
        if case_studies:
            competitors_html = self._build_competitors_section(case_studies)
            sections.append(f'<div class="page-break"></div>{competitors_html}')

        # Gap Analysis section (page break)
        if gap_analysis:
            gaps_html = self._build_gap_analysis_section(gap_analysis)
            sections.append(f'<div class="page-break"></div>{gaps_html}')

        # Inside Intel section (page break)
        if internal_ops:
            intel_html = self._build_inside_intel_section(internal_ops)
            sections.append(f'<div class="page-break"></div>{intel_html}')

        # Sources section (page break)
        if report and report.web_sources:
            sources_html = self._build_sources_section(report.web_sources)
            sections.append(f'<div class="page-break"></div>{sources_html}')

        # Raw Output section (page break)
        if research_job.result:
            raw_html = self._build_raw_output_section(research_job.result)
            sections.append(f'<div class="page-break"></div>{raw_html}')

        # Combine all sections
        content = '\n'.join(sections)

        return self.RESEARCH_REPORT_TEMPLATE.format(
            title=f"Research Report: {research_job.client_name}",
            content=content,
            generated_date=datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        )

    def _build_header(self, job) -> str:
        """Build the report header."""
        vertical_html = ''
        if job.vertical:
            vertical_display = job.vertical.replace('_', ' ')
            vertical_html = f'<div class="vertical-badge">{html.escape(vertical_display)}</div>'

        return f'''
        <div class="header">
            <h1>{html.escape(job.client_name)}</h1>
            {vertical_html}
            <div class="generated-date">Generated on {datetime.now().strftime('%B %d, %Y')}</div>
        </div>
        '''

    def _build_overview_section(self, job, report) -> str:
        """Build the Overview section."""
        sections = ['<div class="section"><div class="section-title">Overview</div>']

        # Stats grid
        if report:
            stats = []
            if report.founded_year:
                stats.append(f'<div class="stat-card"><div class="label">Founded</div><div class="value">{report.founded_year}</div></div>')
            if report.employee_count:
                stats.append(f'<div class="stat-card"><div class="label">Employees</div><div class="value">{html.escape(report.employee_count)}</div></div>')
            if report.annual_revenue:
                stats.append(f'<div class="stat-card"><div class="label">Revenue</div><div class="value">{html.escape(report.annual_revenue)}</div></div>')
            if report.digital_maturity:
                maturity_display = report.digital_maturity.replace('_', ' ').title()
                stats.append(f'<div class="stat-card"><div class="label">Digital Maturity</div><div class="value">{html.escape(maturity_display)}</div></div>')

            if stats:
                sections.append(f'<div class="stats-grid">{"".join(stats)}</div>')

            # Company Overview
            if report.company_overview:
                sections.append(f'''
                <div class="subsection-title">Company Overview</div>
                <p>{html.escape(report.company_overview)}</p>
                ''')

            # Decision Makers
            if report.decision_makers:
                sections.append('<div class="subsection-title">Key Decision Makers</div>')
                for dm in report.decision_makers:
                    name = html.escape(dm.get('name') or '')
                    title = html.escape(dm.get('title') or '')
                    background = html.escape(dm.get('background') or '')
                    sections.append(f'''
                    <div class="card">
                        <div class="card-title">{name}</div>
                        <div class="card-subtitle">{title}</div>
                        {f'<div class="card-content">{background}</div>' if background else ''}
                    </div>
                    ''')

            # Pain Points
            if report.pain_points:
                sections.append('<div class="subsection-title">Pain Points</div><ul>')
                for point in report.pain_points:
                    sections.append(f'<li style="color: #c62828;">{html.escape(point)}</li>')
                sections.append('</ul>')

            # Opportunities
            if report.opportunities:
                sections.append('<div class="subsection-title">Opportunities</div><ul>')
                for opp in report.opportunities:
                    sections.append(f'<li style="color: #2e7d32;">{html.escape(opp)}</li>')
                sections.append('</ul>')

            # Talking Points
            if report.talking_points:
                sections.append('<div class="subsection-title">Recommended Talking Points</div>')
                for point in report.talking_points:
                    sections.append(f'<div class="highlight-box">{html.escape(point)}</div>')
        else:
            sections.append('<p>Overview data not available.</p>')

        sections.append('</div>')
        return '\n'.join(sections)

    def _build_deep_research_section(self, report) -> str:
        """Build the Deep Research section."""
        sections = ['<div class="section"><div class="section-title">Deep Research</div>']

        # Company Details
        details = []
        if report.headquarters:
            details.append(('Headquarters', report.headquarters))
        if report.website:
            details.append(('Website', report.website))
        if report.founded_year:
            details.append(('Founded', str(report.founded_year)))
        if report.employee_count:
            details.append(('Employees', report.employee_count))
        if report.annual_revenue:
            details.append(('Revenue', report.annual_revenue))

        if details:
            sections.append('<div class="subsection-title">Company Details</div>')
            sections.append('<table><tr><th>Field</th><th>Value</th></tr>')
            for label, value in details:
                sections.append(f'<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>')
            sections.append('</table>')

        # AI Assessment
        if report.digital_maturity or report.ai_adoption_stage or report.ai_footprint:
            sections.append('<div class="subsection-title">Digital & AI Assessment</div>')
            assessment_items = []
            if report.digital_maturity:
                maturity_display = report.digital_maturity.replace('_', ' ').title()
                assessment_items.append(f'<div class="stat-card"><div class="label">Digital Maturity</div><div class="value">{html.escape(maturity_display)}</div></div>')
            if report.ai_adoption_stage:
                stage_display = report.ai_adoption_stage.replace('_', ' ').title()
                assessment_items.append(f'<div class="stat-card"><div class="label">AI Adoption</div><div class="value">{html.escape(stage_display)}</div></div>')
            if assessment_items:
                sections.append(f'<div class="stats-grid">{"".join(assessment_items)}</div>')
            if report.ai_footprint:
                sections.append(f'<p>{html.escape(report.ai_footprint)}</p>')

        # Recent News
        if report.recent_news:
            sections.append('<div class="subsection-title">Recent News</div>')
            for news in report.recent_news[:5]:  # Limit to 5 items
                title = html.escape(news.get('title') or '')
                summary = html.escape(news.get('summary') or '')
                date = html.escape(news.get('date') or '')
                source = html.escape(news.get('source') or '')
                sections.append(f'''
                <div class="card">
                    <div class="card-title">{title}</div>
                    <div class="card-content">{summary}</div>
                    <div class="card-subtitle">{date}{f" - {source}" if source else ""}</div>
                </div>
                ''')

        # Strategic Goals
        if report.strategic_goals:
            sections.append('<div class="subsection-title">Strategic Goals</div><ol>')
            for goal in report.strategic_goals:
                sections.append(f'<li>{html.escape(goal)}</li>')
            sections.append('</ol>')

        # Key Initiatives
        if report.key_initiatives:
            sections.append('<div class="subsection-title">Key Initiatives</div>')
            for init in report.key_initiatives:
                sections.append(f'<div class="highlight-box">{html.escape(init)}</div>')

        sections.append('</div>')
        return '\n'.join(sections)

    def _build_competitors_section(self, case_studies) -> str:
        """Build the Competitors section."""
        sections = ['<div class="section"><div class="section-title">Competitor Case Studies</div>']
        sections.append(f'<p>Found {len(case_studies)} relevant competitor case studies:</p>')

        sections.append('''
        <table>
            <tr>
                <th>Competitor</th>
                <th>Case Study</th>
                <th>Technologies</th>
                <th>Match</th>
            </tr>
        ''')

        for cs in case_studies:
            competitor_name = html.escape(cs.competitor_name)
            case_title = html.escape(cs.case_study_title)
            summary = html.escape(cs.summary[:200] + '...' if len(cs.summary) > 200 else cs.summary)
            technologies = ', '.join(cs.technologies_used[:3]) if cs.technologies_used else '-'
            match_pct = f"{round(cs.relevance_score * 100)}%"

            sections.append(f'''
            <tr>
                <td><strong>{competitor_name}</strong></td>
                <td>{case_title}<br><small style="color: #666;">{summary}</small></td>
                <td>{html.escape(technologies)}</td>
                <td style="text-align: center;"><strong>{match_pct}</strong></td>
            </tr>
            ''')

        sections.append('</table>')

        # Outcomes for top case studies
        for cs in case_studies[:3]:
            if cs.outcomes:
                sections.append(f'<div class="subsection-title">{html.escape(cs.competitor_name)} - Key Outcomes</div>')
                sections.append('<ul>')
                for outcome in cs.outcomes:
                    sections.append(f'<li style="color: #2e7d32;">{html.escape(outcome)}</li>')
                sections.append('</ul>')

        sections.append('</div>')
        return '\n'.join(sections)

    def _build_gap_analysis_section(self, gap_analysis) -> str:
        """Build the Gap Analysis section."""
        sections = ['<div class="section"><div class="section-title">Gap Analysis</div>']

        # Confidence score
        confidence_pct = round(gap_analysis.confidence_score * 100)
        confidence_class = 'confidence-high' if confidence_pct >= 70 else ('confidence-medium' if confidence_pct >= 40 else 'confidence-low')
        sections.append(f'''
        <p>Analysis Confidence: <span class="confidence-badge {confidence_class}">{confidence_pct}%</span></p>
        ''')

        # Priority Areas
        if gap_analysis.priority_areas:
            sections.append('<div class="subsection-title">Priority Areas</div>')
            for i, area in enumerate(gap_analysis.priority_areas, 1):
                sections.append(f'<div class="warning-box"><strong>#{i}</strong> {html.escape(area)}</div>')

        # Gap Grid
        has_gaps = gap_analysis.technology_gaps or gap_analysis.capability_gaps or gap_analysis.process_gaps
        if has_gaps:
            sections.append('<div class="subsection-title">Identified Gaps</div><div class="gap-grid">')

            if gap_analysis.technology_gaps:
                sections.append('<div class="gap-column"><strong>Technology Gaps</strong>')
                for gap in gap_analysis.technology_gaps:
                    sections.append(f'<div class="gap-item gap-tech">{html.escape(gap)}</div>')
                sections.append('</div>')

            if gap_analysis.capability_gaps:
                sections.append('<div class="gap-column"><strong>Capability Gaps</strong>')
                for gap in gap_analysis.capability_gaps:
                    sections.append(f'<div class="gap-item gap-capability">{html.escape(gap)}</div>')
                sections.append('</div>')

            if gap_analysis.process_gaps:
                sections.append('<div class="gap-column"><strong>Process Gaps</strong>')
                for gap in gap_analysis.process_gaps:
                    sections.append(f'<div class="gap-item gap-process">{html.escape(gap)}</div>')
                sections.append('</div>')

            sections.append('</div>')

        # Recommendations
        if gap_analysis.recommendations:
            sections.append('<div class="subsection-title">Recommendations</div>')
            for rec in gap_analysis.recommendations:
                sections.append(f'<div class="success-box">{html.escape(rec)}</div>')

        # Analysis Notes
        if gap_analysis.analysis_notes:
            sections.append(f'<div class="subsection-title">Analysis Notes</div><p>{html.escape(gap_analysis.analysis_notes)}</p>')

        sections.append('</div>')
        return '\n'.join(sections)

    def _build_inside_intel_section(self, intel) -> str:
        """Build the Inside Intel section."""
        sections = ['<div class="section"><div class="section-title">Inside Intel</div>']

        # Employee Sentiment
        if intel.employee_sentiment:
            sentiment = intel.employee_sentiment
            sections.append('<div class="subsection-title">Employee Sentiment</div>')

            # Overall rating
            overall = sentiment.get('overall_rating', 0)
            sections.append(f'''
            <div class="stats-grid">
                <div class="stat-card"><div class="label">Overall Rating</div><div class="value">{overall:.1f}/5.0</div></div>
                <div class="stat-card"><div class="label">Would Recommend</div><div class="value">{sentiment.get("recommend_pct", 0)}%</div></div>
                <div class="stat-card"><div class="label">Trend</div><div class="value" style="text-transform: capitalize;">{sentiment.get("trend", "stable")}</div></div>
            </div>
            ''')

            # Category ratings
            categories = ['work_life_balance', 'compensation', 'culture', 'management']
            category_ratings = []
            for cat in categories:
                val = sentiment.get(cat, 0)
                if val:
                    label = cat.replace('_', ' ').title()
                    category_ratings.append(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{val:.1f}</div></div>')
            if category_ratings:
                sections.append(f'<div class="stats-grid">{"".join(category_ratings)}</div>')

            # Themes
            positive_themes = sentiment.get('positive_themes', [])
            negative_themes = sentiment.get('negative_themes', [])
            if positive_themes:
                themes_html = ', '.join(html.escape(t) for t in positive_themes)
                sections.append(f'<p><strong>Positive Themes:</strong> {themes_html}</p>')
            if negative_themes:
                themes_html = ', '.join(html.escape(t) for t in negative_themes)
                sections.append(f'<p><strong>Negative Themes:</strong> {themes_html}</p>')

        # Job Postings / Hiring
        if intel.job_postings:
            jobs = intel.job_postings
            sections.append('<div class="subsection-title">Hiring Intelligence</div>')
            sections.append(f'''
            <div class="stats-grid">
                <div class="stat-card"><div class="label">Total Open Positions</div><div class="value">{jobs.get("total_openings", 0)}</div></div>
            </div>
            ''')

            # Departments hiring
            depts = jobs.get('departments_hiring', {})
            if depts:
                sections.append('<p><strong>Departments Hiring:</strong></p><ul>')
                for dept, count in sorted(depts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    sections.append(f'<li>{html.escape(dept)}: {count} positions</li>')
                sections.append('</ul>')

            # Skills sought
            skills = jobs.get('skills_sought', [])
            if skills:
                skills_html = ', '.join(html.escape(s) for s in skills[:10])
                sections.append(f'<p><strong>Key Skills Sought:</strong> {skills_html}</p>')

            # Urgency signals
            urgency = jobs.get('urgency_signals', [])
            if urgency:
                sections.append('<p><strong>Urgency Signals:</strong></p>')
                for signal in urgency:
                    sections.append(f'<div class="warning-box">{html.escape(signal)}</div>')

            # Insights
            insights = jobs.get('insights', '')
            if insights:
                sections.append(f'<div class="highlight-box">{html.escape(insights)}</div>')

        # LinkedIn Presence
        if intel.linkedin_presence:
            linkedin = intel.linkedin_presence
            sections.append('<div class="subsection-title">LinkedIn Presence</div>')
            sections.append(f'''
            <div class="stats-grid">
                <div class="stat-card"><div class="label">Followers</div><div class="value">{linkedin.get("follower_count", 0):,}</div></div>
                <div class="stat-card"><div class="label">Engagement</div><div class="value" style="text-transform: capitalize;">{linkedin.get("engagement_level", "n/a")}</div></div>
                <div class="stat-card"><div class="label">Employee Trend</div><div class="value" style="text-transform: capitalize;">{linkedin.get("employee_trend", "stable")}</div></div>
            </div>
            ''')

            # Notable changes
            changes = linkedin.get('notable_changes', [])
            if changes:
                sections.append('<p><strong>Notable Changes:</strong></p><ul>')
                for change in changes:
                    sections.append(f'<li>{html.escape(change)}</li>')
                sections.append('</ul>')

        # Key Insights
        if intel.key_insights:
            sections.append('<div class="subsection-title">Key Insights & Recommendations</div>')
            for insight in intel.key_insights:
                sections.append(f'<div class="warning-box">{html.escape(insight)}</div>')

        # Gap Correlations
        if intel.gap_correlations:
            sections.append('<div class="subsection-title">Gap Correlation Insights</div>')
            sections.append('''
            <table>
                <tr>
                    <th>Gap Type</th>
                    <th>Description</th>
                    <th>Evidence</th>
                    <th>Confidence</th>
                </tr>
            ''')
            for corr in intel.gap_correlations:
                gap_type = html.escape((corr.get('gap_type') or '').title())
                desc = html.escape(corr.get('description') or '')
                evidence = html.escape(corr.get('evidence') or '')
                confidence = f"{round(corr.get('confidence', 0) * 100)}%"
                sections.append(f'''
                <tr>
                    <td><strong>{gap_type}</strong></td>
                    <td>{desc}</td>
                    <td>{evidence}</td>
                    <td style="text-align: center;">{confidence}</td>
                </tr>
                ''')
            sections.append('</table>')

        # Confidence footer
        confidence_pct = round(intel.confidence_score * 100)
        confidence_class = 'confidence-high' if confidence_pct >= 70 else ('confidence-medium' if confidence_pct >= 40 else 'confidence-low')
        freshness = intel.data_freshness.replace('_', ' ') if intel.data_freshness else 'unknown'
        sections.append(f'''
        <p style="margin-top: 15px;">
            Confidence: <span class="confidence-badge {confidence_class}">{confidence_pct}%</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;Data Freshness: {html.escape(freshness)}
        </p>
        ''')

        sections.append('</div>')
        return '\n'.join(sections)

    def _build_sources_section(self, sources) -> str:
        """Build the Sources section."""
        sections = ['<div class="section"><div class="section-title">Sources</div>']
        sections.append(f'<p>Research grounded with {len(sources)} web source{"s" if len(sources) != 1 else ""}:</p>')

        for i, source in enumerate(sources, 1):
            title = html.escape(source.get('title') or 'Untitled Source')
            uri = source.get('uri') or ''
            uri_display = html.escape(uri) if uri else ''

            sections.append(f'''
            <div class="source-item">
                <div class="source-number">{i}</div>
                <div class="source-content">
                    <div class="source-title">{title}</div>
                    {f'<div class="source-url">{uri_display}</div>' if uri_display else ''}
                </div>
            </div>
            ''')

        sections.append('<p style="font-size: 10px; color: #666; margin-top: 15px;">Sources collected via Google Search grounding for real-time, verified information.</p>')
        sections.append('</div>')
        return '\n'.join(sections)

    def _build_raw_output_section(self, result) -> str:
        """Build the Raw Output section with full content."""
        sections = ['<div class="section"><div class="section-title">Raw Output</div>']
        sections.append(f'<div class="raw-output">{html.escape(result)}</div>')
        sections.append('</div>')
        return '\n'.join(sections)
