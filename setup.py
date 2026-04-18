"""
Smart Academic Notes - Infrastructure Setup & Verification Engine

This utility script automates the environment initialization for the Smart Academic Notes
Generator. It performs dependency validation, environment variable configuration,
directory structure creation, and service availability checks to ensure a seamless
onboarding experience for developers.

Functions:
    check_python_version: Validates the runtime environment.
    install_requirements: Handles dependency resolution.
    create_env_file: Initializes local configuration.
    download_nltk_data: Fetches NLP resources.
    check_services: Validates API integrations.
    create_project_structure: Bootstraps local storage.
    test_imports: Verifies package integrity.
"""

import os
import sys
import subprocess


def check_python_version():
    """
    Verifies that the current Python interpreter meets the minimum version requirements.

    Smart Academic Notes requires Python 3.8+ to leverage modern async features
    and type hinting capabilities.

    Raises:
        SystemExit: If the Python version is below the 3.8 threshold.
    """
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")


def install_requirements():
    """
    Automates the installation of project dependencies via pip.

    Iterates through the requirements.txt file and ensures all necessary
    libraries for AI processing, database interaction, and PDF generation are present.
    """
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)


def create_env_file():
    """
    Initializes the local environment configuration file (.env).

    If a .env file does not exist, this function clones the .env.example template
    or creates a default one with placeholders for essential API keys (Supabase, 
    Google Gemini, Cloudinary, etc.).
    """
    if not os.path.exists(".env"):
        print("📝 Creating .env file...")

        # Copy from example
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as f:
                content = f.read()
            with open(".env", "w") as f:
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
            with open(".env", "w") as f:
                f.write(env_content)

        print("✅ .env file created")
        print("⚠️  Please update .env with your actual API keys and credentials")
    else:
        print("✅ .env file already exists")


def download_nltk_data():
    """
    Downloads essential Natural Language Toolkit (NLTK) corpora and models.

    Specifically fetches tokenizers and stopword lists required for the 
    semantic chunking and analysis engine.
    """
    print("📚 Downloading NLTK data...")
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Failed to download NLTK data: {e}")


def check_services():
    """
    Validates the presence and validity of integrated third-party services.

    Performs a verification check on Supabase, Google Gemini, and Cloudinary 
    credentials found in the environment variables to pre-emptively identify 
    configuration gaps.

    Returns:
        bool: True if all core services are detected as configured, False otherwise.
    """
    print("\n🔍 Checking service configurations...")

    from dotenv import load_dotenv

    load_dotenv()

    services_status = {
        "Supabase": bool(
            os.getenv("SUPABASE_URL")
            and "your-project" not in os.getenv("SUPABASE_URL", "")
        ),
        "Google Gemini": bool(
            os.getenv("GOOGLE_API_KEY")
            and "your-google" not in os.getenv("GOOGLE_API_KEY", "")
        ),
        "Cloudinary": bool(
            os.getenv("CLOUDINARY_CLOUD_NAME")
            and "your-cloud" not in os.getenv("CLOUDINARY_CLOUD_NAME", "")
        ),
        "JWT Secret": bool(
            os.getenv("JWT_SECRET") and "your-super" not in os.getenv("JWT_SECRET", "")
        ),
    }

    for service, configured in services_status.items():
        status = "✅" if configured else "❌"
        print(f"{status} {service}: {'Configured' if configured else 'Not configured'}")

    unconfigured = [
        service for service, configured in services_status.items() if not configured
    ]

    if unconfigured:
        print("\n⚠️  Please configure the following services in your .env file:")
        for service in unconfigured:
            print(f"   - {service}")
        print("\nRefer to README.md for setup instructions")
        return False

    return True


def create_project_structure():
    """
    Ensures the necessary directory hierarchy exists on the local filesystem.

    Strips out local storage paths for PDFs, audio chunks, and logs to prevent 
    IOErrors during runtime processing.
    """
    directories = ["storage", "storage/pdfs", "storage/audio", "logs"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")


def test_imports():
    """
    Performs a smoke test by attempting to import all critical project dependencies.

    This ensures that the virtual environment is correctly configured and all 
    native extensions (like PyMuPDF or FAISS) are functional on the current OS.

    Returns:
        bool: True if all imports pass, False if any critical dependency is missing.
    """
    print("\n🧪 Testing package imports...")

    required_packages = [
        "flask",
        "google.generativeai",
        "supabase",
        "cloudinary",
        "PyPDF2",
        "langchain",
        "faiss",
        "jwt",
        "speech_recognition",
        "nltk",
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
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
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
    print("=" * 60)

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
