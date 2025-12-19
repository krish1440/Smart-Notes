import os
from typing import List
from dotenv import load_dotenv
import jwt
import json
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, Response, render_template, request, jsonify, session
from flask_cors import CORS
import logging
from threading import Timer
import markdown
import uuid
import time
import warnings
import httpx
from retry import retry
import postgrest.exceptions
import fitz  #  for PDF processing
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from bs4 import BeautifulSoup
from io import BytesIO

# Configure logiing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_pinecone")
warnings.filterwarnings("ignore", message=".*pydantic.*")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
CORS(app)
load_dotenv()


try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
    JWT_SECRET = os.getenv("JWT_SECRET")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY, CLOUDINARY_CLOUD_NAME, 
                CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, JWT_SECRET, PINECONE_API_KEY, 
                PINECONE_INDEX_NAME, HUGGINGFACE_API_KEY]):
        raise ValueError("Missing required environment variables")
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {str(e)}")
    raise

# Lazy Initializer for services
class LazyInitializer:
    def __init__(self):
        self._supabase = None
        self._pinecone_client = None
        self._pinecone_index = None
        self._cloudinary = None
        self._hf_embeddings = None
        self._llm = None
        self._nltk_initialized = False

    def get_supabase(self):
        if self._supabase is None:
            from supabase import create_client, Client
            try:
                self._supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase client initialized")
            except Exception as e:
                logger.error(f"Error initializing Supabase: {str(e)}")
                raise
        return self._supabase

    def get_cloudinary(self):
        if self._cloudinary is None:
            import cloudinary
            import cloudinary.uploader
            try:
                cloudinary.config(
                    cloud_name=CLOUDINARY_CLOUD_NAME,
                    api_key=CLOUDINARY_API_KEY,
                    api_secret=CLOUDINARY_API_SECRET
                )
                self._cloudinary = cloudinary
                logger.info("Cloudinary client initialized")
            except Exception as e:
                logger.error(f"Error initializing Cloudinary: {str(e)}")
                raise
        return self._cloudinary

    def get_pinecone_client(self):
        if self._pinecone_client is None:
            from pinecone import Pinecone
            try:
                self._pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
                logger.info("Pinecone client initialized")
            except Exception as e:
                logger.error(f"Error initializing Pinecone: {str(e)}")
                raise
        return self._pinecone_client

    def get_pinecone_index(self):
        if self._pinecone_index is None:
            from pinecone import ServerlessSpec
            pc = self.get_pinecone_client()
            try:
                existing_indexes = pc.list_indexes()
                index_names = [index['name'] for index in existing_indexes]
                if PINECONE_INDEX_NAME not in index_names:
                    logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=384,
                        metric='cosine',
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                    time.sleep(10)
                    logger.info(f"Index {PINECONE_INDEX_NAME} created successfully")
                else:
                    logger.info(f"Index {PINECONE_INDEX_NAME} already exists")
                self._pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            except Exception as e:
                if "ALREADY_EXISTS" in str(e):
                    logger.info(f"Index {PINECONE_INDEX_NAME} already exists, proceeding with existing index")
                    self._pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                else:
                    logger.error(f"Error initializing Pinecone index: {str(e)}")
                    raise
        return self._pinecone_index

    def get_hf_embeddings(self):
        if self._hf_embeddings is None:
            from huggingface_hub import InferenceClient
            class HFEmbeddings:
                def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                    self.client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
                    self.model_name = model_name

                @retry(httpx.HTTPError, tries=3, delay=2, backoff=2)
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    try:
                        if not texts:
                            return []
                        response = self.client.feature_extraction(texts, model=self.model_name)
                        return response.tolist()
                    except Exception as e:
                        logger.error(f"Error generating embeddings via API: {str(e)}")
                        return []

                def embed_query(self, text: str) -> List[float]:
                    embedding = self.embed_documents([text])
                    return embedding[0] if embedding else []

            try:
                self._hf_embeddings = HFEmbeddings()
                logger.info("Hugging Face embeddings initialized")
            except Exception as e:
                logger.error(f"Error initializing HF embeddings: {str(e)}")
                raise
        return self._hf_embeddings

    def get_llm(self):
        if self._llm is None:
            import google.generativeai as genai
            from langchain_google_genai import GoogleGenerativeAI
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self._llm = GoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=GOOGLE_API_KEY)
                logger.info("Google Generative AI initialized")
            except Exception as e:
                logger.error(f"Error initializing Google Generative AI: {str(e)}")
                raise
        return self._llm

    def initialize_nltk(self):
        if not self._nltk_initialized:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            self._nltk_initialized = True
            logger.info("NLTK data initialized")

#  loader
lazy_init = LazyInitializer()

# Timeout decorator
def timeout(seconds):
    def decorator(func):
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = Timer(seconds, lambda: _timeout_handler(None, None))
            timer.start()
            try:
                result = func(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return wrapper
    return decorator

# Rate limiting helper  ( timeout )
@timeout(60)
def rate_limit_llm(func, *args, max_retries=3, delay=2):
    import google.generativeai as genai as genai
    for attempt in range(max_retries):
        try:
            return func(*args)
        except genai.types.BlockedPromptException as e:
            logger.error(f"LLM blocked prompt error: {str(e)}")
            return None
        except genai.types.ResponseError as e:
            if "429" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"LLM rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached. LLM operation failed: {str(e)}")
                    return None
            else:
                logger.error(f"Non-rate-limit LLM error: {str(e)}")
                return None
    return None

def init_db():
    try:
        supabase = lazy_init.get_supabase()
        supabase.table('users').select("*").limit(1).execute()
        logger.info("Database tables verified")
    except postgrest.exceptions.APIError as e:
        logger.error(f"Error verifying database tables: {str(e)}")
        raise

# Utility functions
def generate_token(user_id, email):
    try:
        payload = {
            'user_id': user_id,
            'email': email,
            'exp': datetime.now(timezone.utc) + timedelta(days=7)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    except (TypeError, ValueError) as e:
        logger.error(f"Error generating JWT token: {str(e)}")
        raise

def verify_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid JWT token: {str(e)}")
        return None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            logger.warning("No token provided")
            return jsonify({'error': 'No token provided'}), 401
        
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_token(token)
        if not payload:
            logger.warning("Invalid token")
            return jsonify({'error': 'Invalid token'}), 401
        
        request.current_user = payload
        return f(*args, **kwargs)
    return decorated

@timeout(60)
def extract_text_from_pdf(file_content):
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text("text") or ""
            text += page_text + "\n"
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text with PyMuPDF: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        def stream_chunks():
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                yield from text_splitter.split_text(chunk)
        
        return list(stream_chunks())
    except ValueError as e:
        logger.error(f"Error chunking text: {str(e)}")
        return []

@timeout(60)
def create_pinecone_vectors(chunks, note_id, user_id, batch_size=5):
    pinecone_index = lazy_init.get_pinecone_index()
    hf_embeddings = lazy_init.get_hf_embeddings()
    try:
        namespace = f"user_{user_id}_note_{note_id}"
        vectors = []
        chunk_id = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = hf_embeddings.embed_documents(batch_chunks)
            if not embeddings:
                raise ValueError(f"No embeddings generated for batch starting at index {i}")
            
            for chunk, emb in zip(batch_chunks, embeddings):
                vec_id = f"chunk_{chunk_id}_{uuid.uuid4()}"
                vectors.append({
                    "id": vec_id,
                    "values": emb,
                    "metadata": {
                        'note_id': str(note_id),
                        'user_id': str(user_id),
                        'chunk_id': chunk_id,
                        'text': chunk[:500]
                    }
                })
                chunk_id += 1
            
            pinecone_index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Upserted batch {(i//batch_size) + 1} of {(len(chunks) + batch_size - 1)//batch_size}")
            vectors = []  # Clear batch to free memory
        
        if chunks:
            test_emb = hf_embeddings.embed_query(chunks[0][:500])
            results = pinecone_index.query(
                vector=test_emb,
                top_k=1,
                namespace=namespace,
                include_metadata=True
            )
            if results.matches and abs(results.matches[0].score - 1.0) < 0.1:
                logger.info(f"Verified: Stored vectors in namespace {namespace}")
                return True, namespace
            else:
                logger.warning(f"Verification failed: {len(results.matches)} matches found")
                return False, None
        
        return True, namespace
    except TimeoutError:
        logger.error("Pinecone vector creation timed out after 60 seconds")
        return False, None
    except Exception as e:
        logger.error(f"Error creating Pinecone vectors: {str(e)}")
        return False, None

@timeout(60)
def get_relevant_chunks(note_id, user_id, query, top_k=5):
    pinecone_index = lazy_init.get_pinecone_index()
    hf_embeddings = lazy_init.get_hf_embeddings()
    try:
        namespace = f"user_{user_id}_note_{note_id}"
        query_emb = hf_embeddings.embed_query(query)
        results = pinecone_index.query(
            vector=query_emb,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        chunks = [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
        scores = [match.score for match in results.matches if 'text' in match.metadata]
        logger.info(f"Retrieved {len(chunks)} chunks with scores: {scores}")
        return chunks
    except TimeoutError:
        logger.error("Pinecone query timed out after 60 seconds")
        return []
    except Exception as e:
        logger.error(f"Error retrieving chunks from Pinecone: {str(e)}")
        return []

def delete_pinecone_vectors(note_id, user_id):
    pinecone_index = lazy_init.get_pinecone_index()
    try:
        namespace = f"user_{user_id}_note_{note_id}"
        pinecone_index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted vectors for note {note_id} in namespace {namespace}")
    except Exception as e:
        logger.error(f"Error deleting Pinecone vectors: {str(e)}")

# Markdown to HTML formatting
def markdown_to_html(text):
    try:
        return markdown.markdown(text, extensions=['extra'])
    except Exception as e:
        logger.error(f"Error converting Markdown to HTML: {str(e)}")
        return text

@timeout(120)
def summarize_text(text):
    import google.generativeai as genai as genai
    def generate_summary():
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Create a detailed summary of the following text.
        Include key points, main topics, and important details:
        
        {text[:4000]}
        """
        try:
            response = model.generate_content(prompt)
            summary = response.text
            return markdown_to_html(summary)
        except genai.types.BlockedPromptException as e:
            logger.error(f"Gemini blocked prompt error: {str(e)}")
            return markdown_to_html("Summary unavailable due to content restrictions.")
        except genai.types.ResponseError as e:
            logger.error(f"Gemini summary generation failed: {str(e)}")
            return markdown_to_html("Error generating summary: Limit reached or content error.")
    
    return rate_limit_llm(generate_summary) or "Error generating summary"

@timeout(60)
def upload_to_cloudinary(file_content, resource_type, folder):
    cloudinary = lazy_init.get_cloudinary()
    try:
        upload_result = cloudinary.uploader.upload(
            file_content,
            resource_type=resource_type,
            folder=folder
        )
        return upload_result['secure_url']
    except Exception as e:
        logger.error(f"Cloudinary error: {str(e)}")
        raise

# Retry decorator for Supabase queries
@retry(httpx.ConnectError, tries=3, delay=1, backoff=2)
def check_pdf_count(user_id, twenty_four_hours_ago):
    supabase = lazy_init.get_supabase()
    return supabase.table('notes').select('count').eq('user_id', user_id).eq('file_type', 'pdf').gte('created_at', twenty_four_hours_ago.isoformat()).execute()

@retry(httpx.ConnectError, tries=3, delay=1, backoff=2)
def check_chat_count(user_id, twenty_four_hours_ago):
    supabase = lazy_init.get_supabase()
    return supabase.table('chats').select('count').eq('user_id', user_id).gte('created_at', twenty_four_hours_ago.isoformat()).execute()

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            logger.warning("Missing email or password in signup request")
            return jsonify({'error': 'Email and password required'}), 400
        
        supabase = lazy_init.get_supabase()
        try:
            existing_user = supabase.table('users').select("*").eq('email', email).execute()
            if existing_user.data:
                logger.warning(f"User already exists: {email}")
                return jsonify({'error': 'User already exists'}), 400
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error checking user: {str(e)}")
            return jsonify({'error': f'Database error checking user: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        scheme = 'https' if request.is_secure else 'http'
        host = request.host
        redirect_url = f"{scheme}://{host}/dashboard"
        
        session['signup_email'] = email
        session['signup_password'] = password
        
        try:
            auth_response = supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "email_redirect_to": redirect_url
                }
            })
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error during signup: {str(e)}")
            return jsonify({'error': f'Failed to process signup due to database error: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase during signup: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        if not auth_response.user:
            logger.warning("Failed to initiate signup process")
            return jsonify({'error': 'Failed to initiate signup process'}), 400
            
        logger.info(f"Signup OTP sent for email: {email}")
        return jsonify({
            'message': 'OTP sent to your email',
            'email': email
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during signup: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/auth/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.json
        otp = data.get('otp')
        
        if not otp:
            logger.warning("Missing OTP in verify request")
            return jsonify({'error': 'OTP is required'}), 400
            
        email = session.get('signup_email')
        password = session.get('signup_password')
        
        if not email or not password:
            logger.warning("Session expired during OTP verification")
            return jsonify({'error': 'Session expired. Please try again'}), 400
            
        supabase = lazy_init.get_supabase()
        try:
            verify_response = supabase.auth.verify_otp({
                "email": email,
                "token": otp,
                "type": "signup"
            })
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error during OTP verification: {str(e)}")
            return jsonify({'error': f'Failed to verify OTP due to database error: {str(e)}'}), 400
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase during OTP verification: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        if not verify_response.user:
            logger.warning("Invalid OTP provided")
            return jsonify({'error': 'Invalid OTP'}), 400
            
        user_id = verify_response.user.id
        
        try:
            result = supabase.table('users').insert({
                'id': user_id,
                'email': email,
                'password': password,
                'created_at': datetime.now(timezone.utc).isoformat()
            }).execute()
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error inserting user: {str(e)}")
            return jsonify({'error': f'Failed to create user record: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase inserting user: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        if not result.data:
            logger.warning("Failed to create user record")
            return jsonify({'error': 'Failed to create user record'}), 400
            
        token = generate_token(user_id, email)
        
        session.pop('signup_email', None)
        session.pop('signup_password', None)
        
        logger.info(f"User verified and created: {email}")
        return jsonify({
            'message': 'Account verified successfully',
            'token': token,
            'user': {
                'id': user_id,
                'email': email
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during OTP verification: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            logger.warning("Missing email or password in login request")
            return jsonify({'error': 'Email and password required'}), 400
        
        supabase = lazy_init.get_supabase()
        try:
            result = supabase.table('users').select("*").eq('email', email).execute()
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error during login: {str(e)}")
            return jsonify({'error': f'Failed to process login due to database error: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase during login: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        if not result.data:
            logger.warning(f"User not found: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = result.data[0]
        
        if user['password'] != password:
            logger.warning(f"Invalid password for user: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        token = generate_token(user['id'], user['email'])
        
        logger.info(f"User logged in: {email}")
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {'id': user['id'], 'email': user['email']}
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/upload/pdf', methods=['POST'])
@require_auth
def upload_pdf():
    try:
        logger.info(f"Starting PDF upload for user {request.current_user['user_id']}")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        user_id = request.current_user['user_id']
        
        now = datetime.now(timezone.utc)
        twenty_four_hours_ago = now - timedelta(hours=24)
        reset_time = now + timedelta(hours=24)
        reset_time_str = reset_time.strftime("%d/%m/%Y %H:%M")
        try:
            count_result = check_pdf_count(user_id, twenty_four_hours_ago)
            pdf_count = count_result.data[0]['count'] if count_result.data else 0
            logger.info(f"User {user_id} has uploaded {pdf_count} PDFs in the last 24 hours")
            if pdf_count >= 5:
                logger.warning(f"User {user_id} exceeded daily PDF upload limit")
                return jsonify({
                    'error': f'Daily PDF upload limit of 5 reached. Please try again after {reset_time_str}.'
                }), 429
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error checking PDF count: {str(e)}")
            return jsonify({'error': f'Database error checking upload limit: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        except Exception as e:
            logger.error(f"Unexpected error checking PDF count: {str(e)}")
            return jsonify({'error': f'Unexpected error checking upload limit: {str(e)}'}), 500
        
        file_content = file.read()
        file_size = len(file_content)
        max_size = 5 * 1024 * 1024
        
        if file_size > max_size:
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({'error': f'File too large. Max size: {max_size // (1024*1024)}MB'}), 400
        
        try:
            secure_url = upload_to_cloudinary(file_content, "raw", "smart-notes/pdfs")
            logger.info(f"File uploaded to Cloudinary: {secure_url}")
        except TimeoutError:
            logger.error("Cloudinary upload timed out after 60 seconds")
            return jsonify({'error': 'File upload to Cloudinary timed out'}), 504
        except Exception as e:
            logger.error(f"Cloudinary error during PDF upload: {str(e)}")
            return jsonify({'error': f'Failed to upload PDF to Cloudinary: {str(e)}'}), 500
        
        try:
            text = extract_text_from_pdf(file_content)
            if not text:
                logger.warning("No text extracted from PDF")
                return jsonify({'error': 'Could not extract text from PDF'}), 400
            text = text.replace('\x00', '')
        except TimeoutError:
            logger.error("PDF processing timed out after 60 seconds")
            return jsonify({'error': 'PDF processing timed out'}), 504
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return jsonify({'error': f'Failed to process PDF content: {str(e)}'}), 400
        
        try:
            summary = summarize_text(text)
            if "Error generating summary" in summary:
                logger.warning("Failed to generate summary")
                return jsonify({'error': 'Failed to generate summary'}), 500
            logger.info("Summary generated successfully")
        except TimeoutError:
            logger.error("Summary generation timed out after 120 seconds")
            return jsonify({'error': 'Summary generation timed out'}), 504
        
        supabase = lazy_init.get_supabase()
        try:
            note_data = {
                'user_id': user_id,
                'title': file.filename,
                'content': text,
                'summary': summary,
                'file_url': secure_url,
                'file_type': 'pdf',
                'pinecone_namespace': '',
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            result = supabase.table('notes').insert(note_data).execute()
            note = result.data[0]
            note_id = note['id']
            logger.info(f"Note {note_id} stored in Supabase for user {user_id}")
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error storing note: {str(e)}")
            return jsonify({'error': f'Failed to store PDF note in database: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase storing note: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        lazy_init.initialize_nltk()
        chunks = chunk_text(text)
        
        logger.info(f"Creating vector store with {len(chunks)} chunks using HF embeddings...")
        try:
            success, namespace = create_pinecone_vectors(chunks, note_id, user_id)
        except TimeoutError:
            logger.error("Pinecone vector creation timed out after 60 seconds")
            success, namespace = False, None
        
        if success and namespace:
            try:
                supabase.table('notes').update({
                    'pinecone_namespace': namespace
                }).eq('id', note_id).execute()
                logger.info(f"Successfully created Pinecone vector store for note {note_id}")
            except postgrest.exceptions.APIError as e:
                logger.error(f"Supabase API error updating namespace: {str(e)}")
                return jsonify({'error': f'Failed to update vector store namespace: {str(e)}'}), 500
            except httpx.ConnectError as e:
                logger.error(f"Connection error to Supabase updating namespace: {str(e)}")
                return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        else:
            logger.warning(f"Warning: Could not create vector store for note {note_id}")
        
        logger.info(f"PDF upload completed for user {user_id}, note {note_id}")
        return jsonify({
            'message': 'PDF processed successfully',
            'note_id': note_id,
            'summary': summary,
            'vector_store_created': success
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during PDF upload: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/notes/<note_id>', methods=['DELETE'])
@require_auth
def delete_note(note_id):
    try:
        user_id = request.current_user['user_id']
        supabase = lazy_init.get_supabase()
        
        try:
            note_result = supabase.table('notes').select("*").eq('id', note_id).eq('user_id', user_id).eq('deleted', False).execute()
            if not note_result.data:
                logger.warning(f"Note {note_id} not found for user {user_id}")
                return jsonify({'error': 'Note not found'}), 404
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to retrieve note: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        note = note_result.data[0]
        
        if note.get('file_url'):
            try:
                file_url = note['file_url']
                if 'cloudinary.com' in file_url:
                    url_parts = file_url.split('/')
                    if len(url_parts) >= 7:
                        public_id_with_folder = '/'.join(url_parts[7:])
                        public_id = public_id_with_folder.rsplit('.', 1)[0]
                        resource_type = "raw" if note['file_type'] == 'pdf' else "video"
                        cloudinary = lazy_init.get_cloudinary()
                        cloudinary.uploader.destroy(public_id, resource_type=resource_type)
                        logger.info(f"Deleted file from Cloudinary: {public_id}")
            except Exception as e:
                logger.error(f"Error deleting from Cloudinary: {str(e)}")
        
        try:
            delete_pinecone_vectors(note_id, user_id)
        except Exception as e:
            logger.error(f"Error deleting Pinecone vectors: {str(e)}")
        
        try:
            supabase.table('chats').delete().eq('note_id', note_id).eq('user_id', user_id).execute()
            supabase.table('notes').update({'deleted': True, 'file_url': None, 'pinecone_namespace': None}).eq('id', note_id).eq('user_id', user_id).execute()
            logger.info(f"Note {note_id} and associated chats soft-deleted for user {user_id}")
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error during note soft-deletion: {str(e)}")
            return jsonify({'error': f'Failed to soft-delete note due to database error: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase during soft-deletion: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        return jsonify({'message': 'Note and all associated data deleted successfully from UI'}), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during note deletion: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    try:
        data = request.json
        message = data.get('message')
        note_id = data.get('note_id')
        
        if not message:
            logger.warning("Missing message in chat request")
            return jsonify({'error': 'Message required'}), 400
        
        user_id = request.current_user['user_id']
        supabase = lazy_init.get_supabase()
        
        now = datetime.now(timezone.utc)
        twenty_four_hours_ago = now - timedelta(hours=24)
        reset_time = now + timedelta(hours=24)
        reset_time_str = reset_time.strftime("%d/%m/%Y %H:%M")
        try:
            count_result = check_chat_count(user_id, twenty_four_hours_ago)
            chat_count = count_result.data[0]['count'] if count_result.data else 0
            logger.info(f"User {user_id} has {chat_count} chats in the last 24 hours")
            if chat_count >= 20:
                logger.warning(f"User {user_id} exceeded daily chat limit")
                return jsonify({
                    'error': f'Daily chat limit of 20 queries reached. Please try again after {reset_time_str}.'
                }), 429
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error checking chat count: {str(e)}")
            return jsonify({'error': f'Failed to check chat limit: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase checking chat count: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        except Exception as e:
            logger.error(f"Unexpected error checking chat count: {str(e)}")
            return jsonify({'error': f'Unexpected error checking chat limit: {str(e)}'}), 500
        
        context = ""
        chat_history = []
        relevant_context = ""
        
        if note_id:
            try:
                note_result = supabase.table('notes').select("*").eq('id', note_id).eq('user_id', user_id).eq('deleted', False).execute()
                if note_result.data:
                    context = note_result.data[0]['content']
                    
                    logger.info(f"Attempting to retrieve relevant chunks for note {note_id}...")
                    try:
                        relevant_chunks = get_relevant_chunks(note_id, user_id, message, top_k=5)
                        if relevant_chunks:
                            relevant_context = "\n\n".join(relevant_chunks)
                            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
                        else:
                            logger.info("No relevant chunks retrieved, using fallback context")
                            relevant_context = context[:2000]
                    except TimeoutError:
                        logger.error("Pinecone query timed out after 60 seconds")
                        relevant_context = context[:2000]
                
                try:
                    chat_result = supabase.table('chats').select("user_message,ai_response").eq('user_id', user_id).eq('note_id', note_id).order('created_at', desc=False).limit(5).execute()
                    if chat_result.data:
                        chat_history = chat_result.data
                except postgrest.exceptions.APIError as e:
                    logger.error(f"Supabase API error retrieving chat history: {str(e)}")
                    return jsonify({'error': f'Failed to retrieve chat history: {str(e)}'}), 500
                except httpx.ConnectError as e:
                    logger.error(f"Connection error to Supabase retrieving chat history: {str(e)}")
                    return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
            except postgrest.exceptions.APIError as e:
                logger.error(f"Supabase API error retrieving note: {str(e)}")
                return jsonify({'error': f'Failed to retrieve note: {str(e)}'}), 500
            except httpx.ConnectError as e:
                logger.error(f"Connection error to Supabase retrieving note: {str(e)}")
                return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        history_str = ""
        for chat in chat_history:
            history_str += f"User: {chat['user_message']}\nAI: {chat['ai_response']}\n"
        
        final_context = relevant_context if relevant_context else context
        
        def generate_response():
            import google.generativeai as genai as genai
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
[SYSTEM INSTRUCTIONS]
You are an advanced AI tutor and knowledge navigator with expertise in analyzing and explaining complex topics.
Your responses must be **comprehensive, detailed, and professional**, written in a clear and educational style suitable for academic study materials or structured reports.
[ROLE]
Expert Academic Tutor & Knowledge Navigator
[GOAL]
- Provide in-depth, well-structured explanations of the given query.
- Break down complex concepts into clear, organized sections.
- Maintain a style similar to academic textbooks (professional, detailed, and structured).
[STYLE & TONE]
- Formal, professional, and engaging
- Every response must be **detailed**, with thorough coverage of sub-points
- Use **examples and applications** to strengthen explanations
[FORMATTING & STRUCTURE RULES]
- Write section titles as **plain text headings** with numbering (e.g., 1. Introduction)
- Do NOT use Markdown heading symbols like ## or ###.
- For sub-sections, use **bold text** or indentation
- Use **headings and subheadings** for organization
- Use **bullet points and sub-points**, with detailed explanations (not one-line answers)
- Include **numbered lists** where step-by-step logic is useful
- **MANDATORILY highlight every key term, technical concept, or important word in `single backticks` (e.g., `term`) to emphasize its significance with a code-like background**. Failure to highlight key terms will result in an incomplete response.
- Use ``` (triple backticks) for any code snippets
- Ensure output is **clean and PDF-friendly** (readable structure, logical flow)
- Always format responses with:
    - Numbered sections for major points
    - Bullet points or indented lines for details
    - Highlighted key terms in `backticks`
- Ensure the structure is **PDF-friendly**: no raw Markdown symbols, only clean text formatting.
[DEPTH & DETAIL EXPECTATIONS]
- Main topics: ~250–300 words each
- Sub-points: ~150–200 words each
- Each point should be **expanded and well-explained**, not just mentioned
- Always provide **background, explanation, and practical context**
- Include **examples, analogies, or applications** where possible
[RESPONSE REQUIREMENTS]
- The AI decides the **most appropriate structure** for the given query (not fixed sections).
- Responses must be **self-contained and complete**, covering definitions, details, context, and applications.
- Responses should be long enough to provide true depth, not surface-level explanations.
- Always maintain **clear formatting with headings, lists, and highlighted terms**.
[CONTEXT ANALYSIS]
Relevant Document Context: {final_context}
Previous Conversation History: {history_str}
[USER QUERY]
{message}
[FINAL INSTRUCTIONS]
Handle greting message like (hey , nice ,great,and all other )your response should be like that also 
If the user query is not related to the uploaded document, politely respond with a clear message such as:
'Sorry, this query is not valid for the uploaded document. Please ask something relevant to the document content.
Generate a **detailed, structured, and well-formatted response** to the user’s query.
**MANDATORILY use `backticks` to highlight every key term, technical concept, or important word throughout the response**.
Ensure the response is **professional, clear, and suitable for inclusion in a PDF document**.
"""
            try:
                response = model.generate_content(prompt)
                return markdown_to_html(response.text)
            except genai.types.BlockedPromptException as e:
                logger.error(f"Gemini blocked prompt error: {str(e)}")
                return markdown_to_html("Sorry, the query was blocked due to content restrictions.")
            except genai.types.ResponseError as e:
                logger.error(f"Gemini response generation failed: {str(e)}")
                return markdown_to_html("Error generating response: Limit reached or content error.")
        
        try:
            formatted_response = rate_limit_llm(generate_response)
            if not formatted_response:
                logger.warning("Failed to generate response after retries")
                return jsonify({'error': 'Error generating response'}), 500
        except TimeoutError:
            logger.error("Response generation timed out after 60 seconds")
            return jsonify({'error': 'Response generation timed out'}), 504
        
        try:
            chat_data = {
                'user_id': user_id,
                'note_id': note_id,
                'user_message': message,
                'ai_response': formatted_response,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            supabase.table('chats').insert(chat_data).execute()
            logger.info(f"Chat stored in Supabase for user {user_id}, note {note_id}")
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error storing chat: {str(e)}")
            return jsonify({'error': f'Failed to store chat: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase storing chat: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        logger.info(f"Chat response generated for user {user_id}, note {note_id}")
        return jsonify({'response': formatted_response}), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during chat: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/notes', methods=['GET'])
@require_auth
def get_notes():
    try:
        user_id = request.current_user['user_id']
        supabase = lazy_init.get_supabase()
        try:
            result = supabase.table('notes').select("*").eq('user_id', user_id).eq('deleted', False).order('created_at', desc=True).execute()
            notes = result.data
            logger.info(f"Retrieved {len(notes)} notes for user {user_id}")
            return jsonify({'notes': notes}), 200
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving notes: {str(e)}")
            return jsonify({'error': f'Failed to retrieve notes: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving notes: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
    except Exception as e:
        logger.error(f"Unexpected error during notes retrieval: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/notes/<note_id>/rename', methods=['PUT'])
@require_auth
def rename_note(note_id):
    try:
        user_id = request.current_user['user_id']
        data = request.json
        new_title = data.get('title')
        
        if not new_title:
            logger.warning("Missing new title in rename request")
            return jsonify({'error': 'New title is required'}), 400
            
        supabase = lazy_init.get_supabase()
        try:
            note_result = supabase.table('notes').select("*").eq('id', note_id).eq('user_id', user_id).eq('deleted', False).execute()
            if not note_result.data:
                logger.warning(f"Note {note_id} not found for user {user_id}")
                return jsonify({'error': 'Note not found'}), 404
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to retrieve note: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        try:
            supabase.table('notes').update({'title': new_title}).eq('id', note_id).eq('user_id', user_id).execute()
            logger.info(f"Note {note_id} renamed to {new_title} for user {user_id}")
            return jsonify({'message': 'Note renamed successfully'}), 200
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error renaming note: {str(e)}")
            return jsonify({'error': f'Failed to rename note: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase renaming note: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
    except Exception as e:
        logger.error(f"Unexpected error during note rename: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/chat/history/<note_id>', methods=['GET'])
@require_auth
def get_chat_history(note_id):
    try:
        user_id = request.current_user['user_id']
        supabase = lazy_init.get_supabase()
        try:
            note_result = supabase.table('notes').select("*").eq('id', note_id).eq('user_id', user_id).eq('deleted', False).execute()
            if not note_result.data:
                logger.warning(f"Note {note_id} not found for user {user_id}")
                return jsonify({'error': 'Note not found'}), 404
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to retrieve note: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        try:
            result = supabase.table('chats') \
                .select('user_message, ai_response, created_at') \
                .eq('user_id', user_id) \
                .eq('note_id', note_id) \
                .order('created_at', desc=False) \
                .execute()
            chats = result.data if result.data else []
            for chat in chats:
                chat['ai_response'] = chat.get('ai_response', '')
            logger.info(f"Retrieved {len(chats)} chat messages for user {user_id}, note {note_id}")
            return jsonify({"chats": chats}), 200
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving chat history: {str(e)}")
            return jsonify({'error': f'Failed to fetch chat history: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving chat history: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
    except Exception as e:
        logger.error(f"Unexpected error during chat history retrieval: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/auth/change-password', methods=['POST'])
@require_auth
def change_password():
    try:
        data = request.json
        email = data.get('email')
        old_password = data.get('old_password')
        new_password = data.get('new_password')
        
        if not email or not old_password or not new_password:
            logger.warning("Missing email, old password, or new password in change-password request")
            return jsonify({'error': 'Email, old password, and new password are required'}), 400
        
        if len(new_password) < 6:
            logger.warning("New password too short")
            return jsonify({'error': 'New password must be at least 6 characters'}), 400
        
        supabase = lazy_init.get_supabase()
        try:
            user = supabase.table('users').select('id, email, password').eq('email', email).execute()
            if not user.data or user.data[0]['password'] != old_password:
                logger.warning(f"Invalid email or old password for user: {email}")
                return jsonify({'error': 'Invalid email or old password'}), 401
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error verifying user credentials: {str(e)}")
            return jsonify({'error': f'Failed to verify credentials: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase verifying credentials: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
        
        try:
            supabase.table('users').update({
                'password': new_password
            }).eq('email', email).execute()
            logger.info(f"Password changed successfully for user: {email}")
            return jsonify({'message': 'Password changed successfully'}), 200
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error updating password: {str(e)}")
            return jsonify({'error': f'Failed to update password: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase updating password: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503
    
    except Exception as e:
        logger.error(f"Unexpected error in change-password endpoint: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/notes/<note_id>/export', methods=['GET'])
@require_auth
def export_note_to_pdf(note_id):
    try:
        user_id = request.current_user['user_id']
        logger.info(f"Exporting note {note_id} to PDF for user {user_id}")
        supabase = lazy_init.get_supabase()

        try:
            note_result = supabase.table('notes').select("title, summary").eq('id', note_id).eq('user_id', user_id).eq('deleted', False).execute()
            if not note_result.data:
                logger.warning(f"Note {note_id} not found for user {user_id}")
                return jsonify({'error': 'Note not found'}), 404
            note = note_result.data[0]
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to retrieve note: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving note: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503

        try:
            chat_result = supabase.table('chats').select('user_message, ai_response, created_at') \
                .eq('user_id', user_id).eq('note_id', note_id).order('created_at', desc=False).execute()
            chats = chat_result.data if chat_result.data else []
            logger.info(f"Retrieved {len(chats)} chat messages for note {note_id}")
        except postgrest.exceptions.APIError as e:
            logger.error(f"Supabase API error retrieving chat history: {str(e)}")
            return jsonify({'error': f'Failed to fetch chat history: {str(e)}'}), 500
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Supabase retrieving chat history: {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 503

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=20,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica-Bold'
        )
        subheading_style = ParagraphStyle(
            'SubHeadingStyle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor('#5a6fd8'),
            fontName='Helvetica-Bold'
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#555555'),
            alignment=TA_JUSTIFY,
            leading=16
        )
        question_style = ParagraphStyle(
            'QuestionStyle',
            parent=body_style,
            fontSize=12,
            textColor=colors.HexColor('#667eea'),
            fontName='Helvetica-Oblique',
            spaceBefore=12
        )
        answer_style = ParagraphStyle(
            'AnswerStyle',
            parent=body_style,
            fontSize=12,
            textColor=colors.HexColor('#333333'),
            spaceBefore=6
        )
        code_style = ParagraphStyle(
            'CodeStyle',
            fontName='Courier',
            fontSize=10,
            backColor=colors.HexColor('#f8f9fa'),
            borderPadding=12,
            borderWidth=1,
            borderColor=colors.HexColor('#e1e5e9'),
            leading=18,
            spaceBefore=12,
            spaceAfter=12,
            textColor=colors.HexColor('#333333'),
            spaceShrinkage=0.0,
            allowWidows=0,
            allowOrphans=0
        )
        highlight_style = ParagraphStyle(
            'HighlightStyle',
            parent=body_style,
            backColor=colors.HexColor('#f0f2ff'),
            borderPadding=3,
            borderWidth=1,
            borderColor=colors.HexColor('#667eea')
        )

        story = []

        def html_to_reportlab(tag):
            markup = ''
            for child in tag.contents:
                if isinstance(child, str):
                    markup += child.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elif child.name == 'strong' or child.name == 'b':
                    markup += f'<b>{html_to_reportlab(child)}</b>'
                elif child.name == 'em' or child.name == 'i':
                    markup += f'<i>{html_to_reportlab(child)}</i>'
                elif child.name == 'code':
                    markup += f'<font backcolor="#f0f2ff" color="#6b4800">{html_to_reportlab(child)}</font>'
                elif child.name == 'u':
                    markup += f'<u>{html_to_reportlab(child)}</u>'
                elif child.name == 'strike':
                    markup += f'<strike>{html_to_reportlab(child)}</strike>'
                elif child.name:
                    markup += html_to_reportlab(child)
            return markup

        def parse_html_tag(tag, style, code_style, highlight_style):
            if not tag:
                return None
            if tag.name == 'p':
                marked_up = html_to_reportlab(tag)
                return Paragraph(marked_up, style)
            elif tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(tag.name[1])
                h_style = heading_style if level <= 2 else subheading_style
                marked_up = html_to_reportlab(tag)
                return Paragraph(marked_up, h_style)
            elif tag.name == 'ol' or tag.name == 'ul':
                bullet_type = '1' if tag.name == 'ol' else 'bullet'
                list_items = []
                for li in tag.find_all('li', recursive=False):
                    li_elem = parse_html_tag(li, style, code_style, highlight_style)
                    if li_elem:
                        list_items.append(ListItem(li_elem))
                return ListFlowable(list_items, bulletType=bullet_type, start='disc' if tag.name == 'ul' else 1, leftIndent=20)
            elif tag.name == 'li':
                marked_up = html_to_reportlab(tag)
                return Paragraph(marked_up, style)
            elif tag.name == 'pre':
                code_tag = tag.find('code')
                code_text = (code_tag.text if code_tag else tag.text) or ''
                return Preformatted(code_text, code_style, dedent=0)
            elif tag.name == 'code':
                marked_up = html_to_reportlab(tag)
                return Paragraph(marked_up, highlight_style)
            elif tag.string:
                return Paragraph(tag.string.strip(), style)
            else:
                children = []
                for child in tag.children:
                    child_elem = parse_html_tag(child, style, code_style, highlight_style)
                    if child_elem:
                        if isinstance(child_elem, list):
                            children.extend(child_elem)
                        else:
                            children.append(child_elem)
                return children if children else None

        def parse_html_content(html_content, content_style, code_style, highlight_style):
            if not html_content:
                return [Paragraph("No content available.", content_style)]
            
            soup = BeautifulSoup(html_content, 'html.parser')
            elements = []
            for tag in soup.children:
                elem = parse_html_tag(tag, content_style, code_style, highlight_style)
                if elem:
                    if isinstance(elem, list):
                        elements.extend(elem)
                    else:
                        elements.append(elem)
            return elements

        story.append(Paragraph(note['title'], title_style))
        story.append(Spacer(1, 24))
        story.append(Paragraph("Summary", heading_style))
        story.extend(parse_html_content(note['summary'], body_style, code_style, highlight_style))
        story.append(Spacer(1, 36))
        story.append(Paragraph("Chat History", heading_style))
        if chats:
            for i, chat in enumerate(chats, 1):
                story.append(Paragraph(f"{i}. Question", subheading_style))
                story.append(Paragraph(chat['user_message'], question_style))
                story.append(Spacer(1, 12))
                story.append(Paragraph("Answer", subheading_style))
                story.extend(parse_html_content(chat['ai_response'], answer_style, code_style, highlight_style))
                story.append(Spacer(1, 36))
        else:
            story.append(Paragraph("No chat history available.", body_style))

        doc.build(story)
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        buffer.close()

        logger.info(f"PDF generated successfully for note {note_id}")
        return Response(
            pdf_data,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename={note["title"]}.pdf'}
        )

    except Exception as e:
        logger.error(f"Unexpected error during PDF export: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)








