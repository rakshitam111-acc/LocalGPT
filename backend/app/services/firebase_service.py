"""Firebase & Google Cloud Firestore Integration Service."""

import os
from typing import Any, Dict, List, Optional

_firestore_db = None
_firebase_initialized = False


def init_firebase():
    """Initialize Firebase Admin SDK with project ID or service account credentials."""
    global _firestore_db, _firebase_initialized
    if _firebase_initialized:
        return _firestore_db

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
        project_id = os.getenv("FIREBASE_PROJECT_ID")

        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            _firestore_db = firestore.client()
            _firebase_initialized = True
            print(f"[Firebase] Connected to Firestore with credentials: {cred_path}")
        elif project_id:
            firebase_admin.initialize_app(options={"projectId": project_id})
            _firestore_db = firestore.client()
            _firebase_initialized = True
            print(f"[Firebase] Connected to Firestore with Project ID: {project_id}")
        else:
            print("[Firebase] No credentials found. Using local database (SQLite/PostgreSQL).")
    except Exception as e:
        print(f"[Firebase] Initialization notice: {e}. Falling back to local database.")

    return _firestore_db


class FirestoreManager:
    """Helper methods for syncing with Firestore."""

    @staticmethod
    def is_connected() -> bool:
        return _firestore_db is not None

    @staticmethod
    def save_conversation(conv_id: str, data: Dict[str, Any]):
        if _firestore_db:
            try:
                _firestore_db.collection("conversations").document(conv_id).set(data, merge=True)
            except Exception as e:
                print(f"[Firestore] Error saving conversation: {e}")

    @staticmethod
    def save_message(conv_id: str, msg_id: str, data: Dict[str, Any]):
        if _firestore_db:
            try:
                _firestore_db.collection("conversations").document(conv_id).collection("messages").document(msg_id).set(data)
            except Exception as e:
                print(f"[Firestore] Error saving message: {e}")
