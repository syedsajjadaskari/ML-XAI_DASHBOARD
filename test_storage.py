import os
from google.cloud import storage
import tempfile

def test_storage_setup():
    bucket_name = os.getenv('BUCKET_NAME', 'your-bucket-name')
    
    try:
        # Test client creation
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Test upload
        test_content = b"Hello, this is a test file!"
        blob = bucket.blob("test/test_file.txt")
        blob.upload_from_string(test_content)
        print("✅ Upload test passed")
        
        # Test download
        downloaded_content = blob.download_as_bytes()
        assert downloaded_content == test_content
        print("✅ Download test passed")
        
        # Test delete
        blob.delete()
        print("✅ Delete test passed")
        
        print("🎉 All storage tests passed!")
        
    except Exception as e:
        print(f"❌ Storage test failed: {e}")

if __name__ == "__main__":
    test_storage_setup()
