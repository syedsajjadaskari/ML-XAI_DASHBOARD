"""
Cleanup Handler for Temporary Storage
Manages session cleanup and storage optimization
"""

import streamlit as st
from src.storage_handler import TemporaryStorageHandler, get_session_id
import logging

logger = logging.getLogger(__name__)

def cleanup_session_on_exit():
    """Clean up user session when they leave."""
    if 'data_gcs_path' in st.session_state:
        try:
            session_id = get_session_id()
            temp_storage = TemporaryStorageHandler(st.secrets["GCS_TEMP_BUCKET"])
            temp_storage.cleanup_user_session(session_id)
            logger.info(f"Session cleanup completed for: {session_id}")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")

def show_storage_usage():
    """Show current storage usage to user."""
    if 'data_gcs_path' in st.session_state:
        data_size = st.session_state.get('data_size_mb', 0)
        storage_cost = data_size * 0.020 / 1024  # $0.020 per GB per month
        
        with st.expander("💾 Storage Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Size", f"{data_size:.1f} MB")
            with col2:
                st.metric("Storage Location", "Google Cloud Storage")
            with col3:
                st.metric("Est. Daily Cost", f"${storage_cost * 30:.4f}")
            
            if st.button("🗑️ Clean Up Now"):
                cleanup_session_on_exit()
                st.success("✅ Storage cleaned up!")
                st.rerun()

# Add to your main app.py
import atexit
atexit.register(cleanup_session_on_exit)