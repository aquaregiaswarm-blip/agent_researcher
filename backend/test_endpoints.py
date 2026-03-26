#!/usr/bin/env python3
"""Quick test of API endpoint structure."""
import json
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def test_endpoint(method, endpoint, data=None, expect_code=None):
    """Test an API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    if data:
        data = json.dumps(data).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            result = response.read().decode()
            print(f"  [OK] {method} {endpoint} -> {response.status}")
            try:
                return True, json.loads(result) if result else {}
            except json.JSONDecodeError:
                return True, {"_raw": result[:100]}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if expect_code and e.code == expect_code:
            print(f"  [OK] {method} {endpoint} -> {e.code} (expected)")
            return True, {}
        print(f"  [FAIL] {method} {endpoint} -> {e.code}: {body[:100]}")
        return False, {}


def main():
    print("Testing API Endpoint Structure")
    print("=" * 60)

    results = []

    # Test admin
    print("\n[Admin]")
    results.append(test_endpoint("GET", "/admin/"))

    # Test research endpoints
    print("\n[Research API]")
    results.append(test_endpoint("POST", "/api/research/", {
        "client_name": "Test Co",
        "sales_history": "Test history"
    }))

    # Test memory endpoints
    print("\n[Memory API]")
    results.append(test_endpoint("GET", "/api/memory/profiles/"))
    results.append(test_endpoint("GET", "/api/memory/plays/"))
    results.append(test_endpoint("GET", "/api/memory/entries/"))

    # Test ideation endpoints
    print("\n[Ideation API]")
    results.append(test_endpoint("GET", "/api/ideation/use-cases/"))
    results.append(test_endpoint("GET", "/api/ideation/plays/"))

    # Test assets endpoints
    print("\n[Assets API]")
    results.append(test_endpoint("GET", "/api/assets/personas/"))
    results.append(test_endpoint("GET", "/api/assets/one-pagers/"))
    results.append(test_endpoint("GET", "/api/assets/citations/"))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r[0])
    total = len(results)
    print(f"Results: {passed}/{total} endpoints working")

    if passed == total:
        print("All endpoint structure tests passed!")
    else:
        print("Some endpoints failed - check output above")


if __name__ == "__main__":
    main()
