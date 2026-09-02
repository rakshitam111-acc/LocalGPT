"""Automated verification suite for FastAPI Backend (Phase 3)."""

import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Ensure root workspace is in sys.path for embeddings.py
root_dir = os.path.dirname(backend_dir)
if root_dir not in sys.path:
    sys.path.insert(1, root_dir)

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app.core.database import init_db
init_db()

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def run_tests():
    print("=" * 65)
    print("🚀 LocalGPT Phase 3 - FastAPI Backend Verification Suite")
    print("=" * 65)

    # 1. Healthcheck Test
    print("\n[1/5] Testing /api/health endpoint...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Healthcheck failed: {res.text}"
    print(f"  [OK] Healthcheck status: {res.json()}")

    # 2. Registration & Login Test
    print("\n[2/5] Testing User Registration & Authentication...")
    test_email = "testuser_phase3@localgpt.ai"
    test_password = "SecurePassword123!"

    reg_res = client.post("/api/auth/register", json={
        "email": test_email,
        "password": test_password,
        "full_name": "Test User",
    })
    if reg_res.status_code == 400:  # already registered
        login_res = client.post("/api/auth/login", json={
            "email": test_email,
            "password": test_password,
        })
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"
        auth_data = login_res.json()
    else:
        assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
        auth_data = reg_res.json()

    token = auth_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("  [OK] Authentication successful! JWT Token acquired.")

    # 3. User Me Profile Test
    print("\n[3/5] Testing Protected /api/auth/me & Settings Update...")
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200, f"Get me failed: {me_res.text}"
    print(f"  [OK] Profile retrieved: {me_res.json()['email']}")

    settings_res = client.patch("/api/auth/settings", json={
        "default_model": "llama-3.3-70b-versatile",
        "default_provider": "groq",
    }, headers=headers)
    assert settings_res.status_code == 200, f"Settings update failed: {settings_res.text}"
    print("  [OK] User settings updated successfully.")

    # 4. Conversations CRUD Test
    print("\n[4/5] Testing Conversations CRUD...")
    # Create conversation
    create_conv = client.post("/api/conversations", json={
        "title": "Quantum Computing Chat",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
    }, headers=headers)
    assert create_conv.status_code == 200, f"Create conversation failed: {create_conv.text}"
    conv_id = create_conv.json()["id"]
    print(f"  [OK] Created conversation ID: {conv_id}")

    # List conversations with search
    list_res = client.get("/api/conversations?search=Quantum", headers=headers)
    assert list_res.status_code == 200
    conv_list = list_res.json()
    assert len(conv_list) >= 1, "Should find at least 1 conversation matching search"
    print(f"  [OK] List & search conversations verified ({len(conv_list)} found).")

    # Rename conversation
    patch_res = client.patch(f"/api/conversations/{conv_id}", json={
        "title": "Advanced Quantum Computing",
    }, headers=headers)
    assert patch_res.status_code == 200
    print("  [OK] Renamed conversation successfully.")

    # 5. Model Catalog Listing Test
    print("\n[5/5] Testing Models Catalog...")
    models_res = client.get("/api/models")
    assert models_res.status_code == 200
    models_data = models_res.json()
    print(f"  [OK] Retrieved {len(models_data['models'])} hosted models and {len(models_data['providers'])} providers.")

    print("\n" + "=" * 65)
    print("🎉 ALL FASTAPI BACKEND TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
