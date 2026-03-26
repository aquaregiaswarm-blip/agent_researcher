#!/usr/bin/env python3
"""Test script for the Deep Prospecting Engine API."""
import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def api_request(method, endpoint, data=None):
    """Make an API request."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    if data:
        data = json.dumps(data).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  Error {e.code}: {e.read().decode()}")
        return None


def test_research_workflow():
    """Test the complete research workflow."""
    print("=" * 60)
    print("Testing Deep Prospecting Engine API")
    print("=" * 60)

    # 1. Create a research job
    print("\n1. Creating research job...")
    job_data = {
        "client_name": "Microsoft",
        "sales_history": "Interested in AI-powered solutions for enterprise data management.",
        "prompt": ""
    }

    result = api_request("POST", "/api/research/", job_data)
    if not result:
        print("  FAILED to create research job")
        return

    job_id = result.get("id")
    print(f"  Created job: {job_id}")
    print(f"  Status: {result.get('status')}")

    # 2. Execute the job
    print("\n2. Executing research job...")
    result = api_request("POST", f"/api/research/{job_id}/execute/")
    if not result:
        print("  FAILED to execute research job")
        return

    print(f"  Execution status: {result.get('status')}")

    if result.get("status") == "completed":
        print("\n4. Research completed!")
        print(f"  Vertical: {result.get('vertical', 'N/A')}")

        # Check for structured report
        report = result.get("report")
        if report:
            print(f"  Company Overview: {report.get('company_overview', 'N/A')[:100]}...")
            print(f"  Digital Maturity: {report.get('digital_maturity', 'N/A')}")
            print(f"  AI Adoption: {report.get('ai_adoption_stage', 'N/A')}")
            print(f"  Pain Points: {len(report.get('pain_points', []))}")
            print(f"  Opportunities: {len(report.get('opportunities', []))}")
            print(f"  Decision Makers: {len(report.get('decision_makers', []))}")
            print(f"  Talking Points: {len(report.get('talking_points', []))}")

        # Check competitors
        competitors = result.get("competitor_case_studies", [])
        print(f"  Competitor Case Studies: {len(competitors)}")

        # Check gap analysis
        gaps = result.get("gap_analysis")
        if gaps:
            print(f"  Gap Analysis Confidence: {gaps.get('confidence_score', 0):.1%}")

        # Check for web sources (Google Search Grounding)
        web_sources = result.get("report", {}).get("web_sources", [])
        print(f"  Web Sources (Grounding): {len(web_sources)}")
        if web_sources:
            print("  Sample sources:")
            for source in web_sources[:5]:
                uri = source.get('uri', 'N/A')
                title = source.get('title', 'N/A')
                print(f"    - {title}: {uri}")

        # 5. Test ideation
        print("\n5. Testing ideation...")
        ideation_result = api_request("POST", "/api/ideation/use-cases/generate/", {
            "research_job_id": job_id
        })
        if ideation_result:
            print(f"  Generated {len(ideation_result)} use cases")
            for uc in ideation_result[:2]:
                print(f"    - {uc.get('title')}")

        # 6. Test asset generation
        print("\n6. Testing asset generation...")

        # Generate personas
        personas_result = api_request("POST", "/api/assets/personas/generate/", {
            "research_job_id": job_id
        })
        if personas_result:
            print(f"  Generated {len(personas_result)} personas")
            for p in personas_result:
                print(f"    - {p.get('name')}: {p.get('title')}")

        # Generate one-pager
        one_pager_result = api_request("POST", "/api/assets/one-pagers/generate/", {
            "research_job_id": job_id
        })
        if one_pager_result:
            print(f"  Generated one-pager: {one_pager_result.get('title')}")

        # Generate account plan
        account_plan_result = api_request("POST", "/api/assets/account-plans/generate/", {
            "research_job_id": job_id
        })
        if account_plan_result:
            print(f"  Generated account plan: {account_plan_result.get('title')}")

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    elif result.get("status") == "failed":
        print(f"\n  Research FAILED: {result.get('error')}")
    else:
        print("\n  Research timed out")


if __name__ == "__main__":
    test_research_workflow()
