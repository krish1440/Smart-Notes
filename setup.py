#!/usr/bin/env python3
"""
Setup script for Smart Academic Notes Generator
"""

import os
import sys
import subprocess
import json

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        
        # Copy from example
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as f:
                content = f.read()
            with open('.env', 'w') as f:
                f.write(content)
        else:
            # Create basic .env file
            env_content = """# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# Google Gemini API
GOOGLE_API_KEY=your-google-gemini-api-key

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# JWT Secret
JWT_SECRET=your-super-secret-jwt-key-change-this

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
"""
            with open('.env', 'w') as f:
                f.write(env_content)
        
        print("✅ .env file created")
        print("⚠️  Please update .env with your actual API keys and credentials")
    else:
        print("✅ .env file already exists")

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Failed to download NLTK data: {e}")

def check_services():
    """Check if required services are configured"""
    print("\n🔍 Checking service configurations...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    services_status = {
        'Supabase': bool(os.getenv('SUPABASE_URL') and 'your-project' not in os.getenv('SUPABASE_URL', '')),
        'Google Gemini': bool(os.getenv('GOOGLE_API_KEY') and 'your-google' not in os.getenv('GOOGLE_API_KEY', '')),
        'Cloudinary': bool(os.getenv('CLOUDINARY_CLOUD_NAME') and 'your-cloud' not in os.getenv('CLOUDINARY_CLOUD_NAME', '')),
        'JWT Secret': bool(os.getenv('JWT_SECRET') and 'your-super' not in os.getenv('JWT_SECRET', ''))
    }
    
    for service, configured in services_status.items():
        status = "✅" if configured else "❌"
        print(f"{status} {service}: {'Configured' if configured else 'Not configured'}")
    
    unconfigured = [service for service, configured in services_status.items() if not configured]
    
    if unconfigured:
        print(f"\n⚠️  Please configure the following services in your .env file:")
        for service in unconfigured:
            print(f"   - {service}")
        print("\nRefer to README.md for setup instructions")
        return False
    
    return True

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        'storage',
        'storage/pdfs',
        'storage/audio',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    required_packages = [
        'flask',
        'google.generativeai',
        'supabase',
        'cloudinary',
        'PyPDF2',
        'langchain',
        'faiss',
        'jwt',
        'speech_recognition',
        'nltk'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please check your installation and try running:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Update your .env file with actual API keys:")
    print("   - Supabase: https://supabase.com")
    print("   - Google AI: https://makersuite.google.com")
    print("   - Cloudinary: https://cloudinary.com")
    print("\n2. Set up your Supabase database:")
    print("   - Run SQL commands from supabase_schema.sql in your Supabase dashboard")
    print("\n3. Start the application:")
    print("   python app.py")
    print("\n4. Open your browser and go to:")
    print("   http://localhost:5000")
    print("\n📖 For detailed instructions, see README.md")

def main():
    """Main setup function"""
    print("🚀 Setting up Smart Academic Notes Generator...")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create .env file
    create_env_file()
    
    # Download NLTK data
    download_nltk_data()
    
    # Create project structure
    create_project_structure()
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Check services configuration
    services_configured = check_services()
    
    # Print next steps
    print_next_steps()
    
    if not services_configured:
        print("\n⚠️  Warning: Some services are not configured yet.")
        print("The application will not work properly until all services are set up.")

if __name__ == "__main__":
    main()