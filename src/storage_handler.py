# Add this to src/storage_handler.py
"""
Temporary Storage Handler for Large Files
Handles file upload/download to/from Google Cloud Storage
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from google.cloud import storage
from typing import Optional, Dict, Any
import logging
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TemporaryStorageHandler:
    """Handles temporary file storage in Google Cloud Storage."""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def upload_file(self, uploaded_file, user_session_id: str) -> str:
        """Upload file to GCS and return the file path."""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
            filename = f"{user_session_id}/{timestamp}_{file_hash}_{uploaded_file.name}"
            
            # Upload to GCS
            blob = self.bucket.blob(filename)
            blob.upload_from_file(uploaded_file, rewind=True)
            
            # Set metadata
            blob.metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'session_id': user_session_id,
                'original_name': uploaded_file.name,
                'file_size': str(uploaded_file.size)
            }
            blob.patch()
            
            logger.info(f"File uploaded to GCS: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {e}")
            raise
    
    def download_file(self, gcs_path: str) -> pd.DataFrame:
        """Download file from GCS and return as DataFrame."""
        try:
            blob = self.bucket.blob(gcs_path)
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                blob.download_to_file(temp_file)
                temp_file_path = temp_file.name
            
            # Read file based on extension
            file_extension = gcs_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(temp_file_path)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(temp_file_path)
            elif file_extension == 'parquet':
                df = pd.read_parquet(temp_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            logger.info(f"File downloaded from GCS: {gcs_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading file from GCS: {e}")
            raise
    
    def delete_file(self, gcs_path: str) -> bool:
        """Delete file from GCS."""
        try:
            blob = self.bucket.blob(gcs_path)
            blob.delete()
            logger.info(f"File deleted from GCS: {gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from GCS: {e}")
            return False
    
    def list_user_files(self, user_session_id: str) -> list:
        """List all files for a user session."""
        try:
            blobs = self.bucket.list_blobs(prefix=f"{user_session_id}/")
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files for user {user_session_id}: {e}")
            return []
    
    def cleanup_user_session(self, user_session_id: str):
        """Clean up all files for a user session."""
        try:
            files = self.list_user_files(user_session_id)
            for file_path in files:
                self.delete_file(file_path)
            logger.info(f"Cleaned up session files for user: {user_session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session for user {user_session_id}: {e}")

# Session management
def get_session_id() -> str:
    """Get or create a unique session ID."""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id