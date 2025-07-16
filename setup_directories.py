#!/usr/bin/env python3
"""
Directory Structure Creator
Creates the necessary directory structure for the Modern ML Web Application
"""

import os
from pathlib import Path
import sys

def create_directory_structure():
    """Create the complete directory structure."""
    
    print("🏗️  Creating directory structure for Modern ML Web Application...")
    print("=" * 60)
    
    # Define the directory structure
    directories = [
        "src",
        "pages", 
        "utils",
        "models",
        "data",
        "data/sample_data",
        "static",
        "static/css",
        "static/images",
        "tests",
        ".streamlit",
        "logs"
    ]
    
    # Define files to create
    files_to_create = {
        # __init__.py files
        "src/__init__.py": '"""\nModern PyCaret-Streamlit ML Web Application\nSource modules for data handling, model training, visualization, and predictions\n"""\n\n__version__ = "2.0.0"\n__author__ = "Updated for 2025 Compatibility"\n',
        
        "pages/__init__.py": '"""\nPage modules for the ML Web Application\n"""\n',
        
        "utils/__init__.py": '"""\nUtility modules for configuration, session management, and helpers\n"""\n',
        
        "tests/__init__.py": '"""\nTest modules\n"""\n',
        
        # Configuration files
        ".streamlit/config.toml": """[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
port = 8501
baseUrlPath = ""
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200
maxMessageSize = 200

[browser]
serverAddress = "localhost"
gatherUsageStats = false
serverPort = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
caching = true
showErrorDetails = false

[runner]
magicEnabled = true
installTracer = false
fixMatplotlib = true
postScriptGC = true
fastReruns = true
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv/
.conda/

# IDE
.vscode/
.idea/
*.swp
*.swo
*.sublime-*

# Streamlit
.streamlit/secrets.toml

# Data files
data/uploads/
*.csv
*.xlsx
*.parquet
*.json

# Models
models/*.pkl
models/*.joblib
models/*.h5

# Logs
logs/*.log
*.log

# OS
.DS_Store
Thumbs.db
.directory

# Temporary files
*.tmp
*.temp
*~

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Environment variables
.env
.env.local
.env.production

# Cache
.cache/
.pytest_cache/
""",
        
        "README.md": """# Modern ML Web Application

## 🚀 Quick Start

1. **Create directory structure:**
   ```bash
   python create_directories.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
pycaret-streamlit-app/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── config.yaml              # Configuration
├── src/                      # Core modules
├── pages/                    # Page modules
├── utils/                    # Utilities
├── models/                   # Saved models
└── tests/                    # Tests
```

## 🔧 Features

- ⚡ Lightning-fast ML training (10-30 seconds)
- 📊 Interactive data exploration
- 🔧 Advanced preprocessing
- 🎯 Multiple ML algorithms
- 📈 Model evaluation and visualization
- 🔮 Single and batch predictions

## 📖 Documentation

See `TRAINING_METHODS_COMPARISON.md` for detailed training options.
""",
        
        "logs/.gitkeep": "# Keep this directory\n",
        "models/.gitkeep": "# Keep this directory\n",
        "data/.gitkeep": "# Keep this directory\n"
    }
    
    # Create directories
    print("📁 Creating directories...")
    for directory in directories:
        dir_path = Path(directory)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {e}")
    
    print()
    
    # Create files
    print("📄 Creating files...")
    for file_path, content in files_to_create.items():
        try:
            file_obj = Path(file_path)
            file_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Created: {file_path}")
        except Exception as e:
            print(f"   ❌ Failed to create {file_path}: {e}")
    
    print()
    
    # Verify structure
    print("🔍 Verifying structure...")
    missing_items = []
    
    for directory in directories:
        if not Path(directory).exists():
            missing_items.append(f"Directory: {directory}")
    
    for file_path in files_to_create.keys():
        if not Path(file_path).exists():
            missing_items.append(f"File: {file_path}")
    
    if missing_items:
        print("   ⚠️  Missing items:")
        for item in missing_items:
            print(f"      - {item}")
    else:
        print("   ✅ All items created successfully!")
    
    print()
    print("📋 Next Steps:")
    print("   1. Copy the provided code files into their respective locations")
    print("   2. Install dependencies: pip install -r requirements.txt")
    print("   3. Run the app: streamlit run app.py")
    print()
    
    # Show final structure
    print("📁 Final Project Structure:")
    print_directory_tree(".", max_depth=3)
    
    return len(missing_items) == 0

def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure."""
    if current_depth >= max_depth:
        return
        
    path = Path(path)
    try:
        # Get items, excluding hidden files and common ignore patterns
        items = []
        for p in path.iterdir():
            name = p.name
            # Skip hidden files and common ignore patterns
            if (not name.startswith('.') or name in ['.streamlit']) and \
               name not in ['__pycache__', '.pytest_cache', '.git']:
                items.append(p)
        
        items = sorted(items, key=lambda x: (x.is_file(), x.name.lower()))
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            
            # Add emoji for different file types
            if item.is_dir():
                emoji = "📁"
            elif item.suffix == '.py':
                emoji = "🐍"
            elif item.suffix in ['.md', '.txt']:
                emoji = "📄"
            elif item.suffix in ['.yaml', '.yml', '.toml']:
                emoji = "⚙️"
            else:
                emoji = "📄"
            
            print(f"{prefix}{current_prefix}{emoji} {item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_directory_tree(item, next_prefix, max_depth, current_depth + 1)
    except PermissionError:
        print(f"{prefix}[Permission Denied]")

def check_prerequisites():
    """Check if Python and required modules are available."""
    print("🔧 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("   ❌ Python 3.7+ required")
        return False
    else:
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if we can write files
    try:
        test_file = Path("test_write.tmp")
        test_file.write_text("test")
        test_file.unlink()
        print("   ✅ Write permissions OK")
    except Exception as e:
        print(f"   ❌ Cannot write files: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🚀 Modern ML Web Application Setup")
    print("=" * 50)
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed!")
        return 1
    
    print()
    
    # Create structure
    success = create_directory_structure()
    
    print()
    if success:
        print("🎉 Directory structure created successfully!")
        print("📝 Don't forget to:")
        print("   1. Copy all the provided code files")
        print("   2. Install requirements: pip install -r requirements.txt")
        print("   3. Run: streamlit run app.py")
        return 0
    else:
        print("⚠️  Some items could not be created. Please check permissions and try again.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)