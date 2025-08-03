# Create src/storage_handler.py (simplified version for testing)
"""
Simplified Storage Handler for Testing
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

def get_session_id() -> str:
    """Get or create a unique session ID."""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

class TemporaryStorageHandler:
    """Simplified storage handler for testing."""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        logger.info(f"Storage handler initialized (bucket: {bucket_name})")
        
    def upload_file(self, uploaded_file, user_session_id: str) -> str:
        """Mock upload - just return a path."""
        filename = f"{user_session_id}/{uploaded_file.name}"
        logger.info(f"Mock upload: {filename}")
        return filename
        
    def download_file(self, gcs_path: str) -> pd.DataFrame:
        """Mock download - return sample data."""
        logger.info(f"Mock download: {gcs_path}")
        # Return sample data for testing
        return pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        
    def cleanup_user_session(self, user_session_id: str):
        """Mock cleanup."""
        logger.info(f"Mock cleanup for session: {user_session_id}")