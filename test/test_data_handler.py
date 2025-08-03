# Create test_data_handler.py
import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_handler import DataHandler

def test_data_handler():
    """Test the DataHandler with sample data."""
    
    # Initialize handler
    config = {
        'app': {
            'max_file_size': 200,
            'supported_formats': ['csv', 'xlsx', 'parquet']
        }
    }
    
    handler = DataHandler(config)
    
    # Test sample data loading
    try:
        data = handler.load_sample_data('titanic')
        print(f"✅ Sample data loaded: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_data_handler()