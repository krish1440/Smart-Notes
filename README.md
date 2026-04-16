# 📚 Smart Academic Notes Generator

An AI-powered web application that automatically generates summarized notes, mind maps, and interactive Q&A chatbots from lecture recordings, PDFs, and academic documents.

## ✨ Features

### 🟢 Free Tier
- Upload PDFs & audio files (up to 5MB)
- Basic AI-powered summarization
- Interactive chat with uploaded content
- Visual mind maps
- Limited chat history

### 🔵 Premium Tier
- Upload large files (up to 100MB)
- Advanced detailed summarization
- Extended chat history
- Priority processing
- Enhanced mind maps

## 🛠️ Tech Stack

- **Backend**: Flask, Python
- **AI/ML**: Google Gemini Pro, LangChain, FAISS
- **Database**: Supabase (PostgreSQL)
- **File Storage**: Cloudinary
- **Frontend**: HTML, CSS, JavaScript, D3.js
- **Authentication**: JWT-based auth

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Supabase Account** - [Sign up here](https://supabase.com)
3. **Google AI Studio API Key** - [Get key here](https://makersuite.google.com/app/apikey)
4. **Cloudinary Account** - [Sign up here](https://cloudinary.com)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd smart-notes-generator
```

2. **Run the automated setup**
```bash
python setup.py
```

3. **Manual setup (alternative)**
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

4. **Configure environment variables**

Edit `.env` file with your credentials:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
GOOGLE_API_KEY=your-google-gemini-api-key
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
JWT_SECRET=your-super-secret-jwt-key
```

5. **Set up Supabase database**
   - Go to your Supabase dashboard
   - Navigate to SQL Editor
   - Run the SQL commands from `supabase_schema.sql`

6. **Run the application**
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📁 Project Structure

```
smart-notes-generator/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── setup.py                 # Automated setup script
├── .env.example             # Environment variables template
├── supabase_schema.sql      # Database schema
├── README.md                # This file
└── templates/               # HTML templates
    ├── index.html           # Login/signup page (with CSS/JS)
    └── dashboard.html       # Main dashboard (with CSS/JS)
```

## 🎯 Application Architecture

### **Backend (`app.py`)**
- **All-in-one Flask backend** with complete functionality
- **Google Gemini Pro** integration for AI processing
- **Supabase** database operations
- **Cloudinary** file storage
- **LangChain + FAISS** for document retrieval
- **JWT authentication** with tier management

### **Frontend Templates**
- **`templates/index.html`** - Complete login/signup interface
- **`templates/dashboard.html`** - Full-featured dashboard
- **Self-contained files** with embedded CSS and JavaScript
- **Responsive design** with mobile support

### **Key Features**

#### **File Processing Pipeline**
1. **PDF Upload** → Text extraction → Chunking → Vector storage
2. **Audio Upload** → Speech-to-text → Processing → Storage
3. **AI Processing** → Summarization + Mind map generation
4. **Chat Integration** → RAG-based Q&A with document context

#### **User Management**
- **JWT-based authentication**
- **Free vs Premium tiers**
- **File size limits based on tier**
- **Usage tracking and limitations**

#### **AI-Powered Features**
- **Smart Summarization** - Context-aware content summaries
- **Mind Map Generation** - Visual topic relationships
- **Intelligent Q&A** - Document-aware chat responses
- **Speech Recognition** - Audio lecture transcription

## 🔧 Configuration

### **Service Setup**

#### **1. Supabase Database**
```bash
# Create project at https://supabase.com
# Copy URL and anon key to .env
# Run SQL schema from supabase_schema.sql
```

#### **2. Google Gemini API**
```bash
# Get API key from https://makersuite.google.com
# Add to .env as GOOGLE_API_KEY
```

#### **3. Cloudinary Storage**
```bash
# Create account at https://cloudinary.com
# Get cloud_name, api_key, api_secret
# Add credentials to .env
```

### **Environment Variables**
```env
# Required for functionality
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-anon-key
GOOGLE_API_KEY=your-gemini-api-key
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-cloudinary-key
CLOUDINARY_API_SECRET=your-cloudinary-secret
JWT_SECRET=random-secret-string

# Optional
FLASK_ENV=development
FLASK_DEBUG=True
```

## 🎮 Usage

### **1. User Registration**
- Sign up with email/password
- Choose Free or Premium tier
- Automatic JWT token generation

### **2. File Upload**
- **Drag & drop** or click to upload
- **PDF files** - Academic papers, textbooks
- **Audio files** - Lecture recordings (.mp3, .wav, .m4a)
- **Automatic processing** with progress indicators

### **3. Generated Content**
- **Smart Summary** - AI-generated key points
- **Mind Map** - Interactive D3.js visualization
- **Chat Interface** - Ask questions about content

### **4. Interactive Features**
- **Real-time chat** with document context
- **Persistent history** - Save conversations
- **Multi-document** support with switching
- **Responsive design** - Mobile-friendly

## 🔍 API Endpoints

### **Authentication**
```
POST /api/auth/signup    # Create account
POST /api/auth/login     # User login
```

### **File Management**
```
POST /api/upload/pdf     # Upload & process PDF
POST /api/upload/audio   # Upload & process audio
GET  /api/notes          # Get user's notes
```

### **Chat System**
```
POST /api/chat                    # Send message
GET  /api/chat/history/<note_id>  # Get chat history
```

### **Frontend Routes**
```
GET  /           # Login/signup page
GET  /dashboard  # Main application dashboard
```

## 🚀 Deployment

### **Local Development**
```bash
python app.py
# Access at http://localhost:5000
```

### **Production Deployment**

#### **1. Environment Setup**
```bash
# Set production environment variables
export FLASK_ENV=production
export FLASK_DEBUG=False
```

#### **2. WSGI Server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### **3. Reverse Proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔧 Customization

### **Adding New AI Features**
```python
# In app.py, modify AI functions:
def custom_analysis(text, user_tier):
    model = genai.GenerativeModel('gemini-pro')
    # Add custom prompt
    return model.generate_content(prompt)
```

### **UI Modifications**
```html
<!-- Edit templates/dashboard.html -->
<!-- Modify embedded CSS/JS for styling changes -->
```

### **Database Extensions**
```sql
-- Add to supabase_schema.sql
CREATE TABLE custom_features (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    -- Add custom fields
);
```

## 🛠️ Troubleshooting

### **Common Issues**

#### **1. Database Connection**
```bash
# Check Supabase credentials
# Verify RLS policies are enabled
# Test connection in Supabase dashboard
```

#### **2. File Upload Problems**
```bash
# Verify Cloudinary credentials
# Check file size limits
# Ensure file types are supported
```

#### **3. AI Processing Errors**
```bash
# Validate Google API key
# Check API quotas and limits
# Verify model availability
```

#### **4. Authentication Issues**
```bash
# Check JWT secret configuration
# Verify token expiration settings
# Clear browser localStorage if needed
```

### **Debug Commands**
```bash
# Enable debug mode
export FLASK_DEBUG=True
python app.py

# Check logs for detailed error messages
# Use browser developer tools for frontend debugging
```

## 📊 Performance Optimization

### **Backend Performance**
- **Database indexing** for faster queries
- **Caching strategies** for repeated AI operations
- **Async processing** for file uploads
- **Connection pooling** for database efficiency

### **Frontend Performance**
- **Lazy loading** for large documents
- **Debounced search** for chat input
- **Efficient re-rendering** for mind maps
- **Mobile optimization** with responsive design

## 🔒 Security Features

- **JWT authentication** with secure tokens
- **Row Level Security (RLS)** in Supabase
- **File type validation** and size limits
- **Input sanitization** for all user inputs
- **CORS configuration** for API security
- **Environment variable protection**

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini** for AI capabilities
- **Supabase** for database infrastructure  
- **Cloudinary** for file storage
- **LangChain** for AI orchestration
- **D3.js** for mind map visualizations
- **Flask** for web framework

---

**Built with ❤️ for students and researchers worldwide.**

## 📞 Support

For support and questions:
- **Create an issue** in the repository
- **Check documentation** for setup guides
- **Review troubleshooting** section above
- **Test with sample files** to verify setup

### **Quick Test**
1. Upload a small PDF file
2. Verify summary generation
3. Test chat functionality
4. Check mind map visualization

Your Smart Academic Notes Generator is ready to transform learning! 🎓