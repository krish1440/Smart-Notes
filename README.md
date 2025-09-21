# 🌟 Smart Notes Application
 
*Effortlessly manage, summarize, and interact with your PDF notes using AI-powered features.*

Welcome to the **Smart Notes Application**, a sleek and powerful web platform that transforms how you handle your PDF notes. Whether you're a student cramming for exams, a professional organizing research, or a curious learner exploring new topics, this app makes it easy to upload, summarize, and chat about your documents. With a user-friendly interface and smart AI features, you can store your notes securely, ask questions about them, and export everything as polished PDFs. Dive in to explore its features, setup guide, and more!

---

## 📑 Table of Contents

- [What Is Smart Notes?](#what-is-smart-notes)
- [Key Features](#key-features)
- [Get Started](#get-started)
- [App Structure](#app-structure)
- [Troubleshooting](#troubleshooting)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)
- [Contact Us](#contact-us)

*Click the links above to jump to a section or scroll to explore!*

---

## 🌍 What Is Smart Notes?

The Smart Notes Application is your go-to tool for managing PDF documents with ease. Upload your notes, get instant summaries, and chat with an AI to dive deeper into the content. The app organizes your notes in a clean dashboard, keeps your data secure, and lets you export your work as professional PDFs. It’s like having a personal study assistant that’s available 24/7!

> **Why Choose Smart Notes?**  
> - **Simple to Use**: Intuitive design for all users.  
> - **AI-Powered**: Summarizes and answers questions about your notes.  
> - **Secure**: Protects your data with strong authentication.  
> - **Organized**: Keeps all your notes in one place.

---

## 🚀 Key Features

Here’s what makes Smart Notes shine:

<details>
<summary><strong>👤 Account Management</strong> (Click to expand)</summary>

- **Sign Up**: Create an account with your email and password. Verify it with a one-time code sent to your email.  
- **Log In**: Access your notes securely with your credentials.  
- **Change Password**: Keep your account safe by updating your password anytime.

</details>

<details>
<summary><strong>📄 PDF Management</strong> (Click to expand)</summary>

- **Upload PDFs**: Add PDF files (up to 5MB) with a daily limit of 5 uploads.  
- **Auto-Summaries**: Get concise summaries of your PDFs to grasp key points quickly.  
- **Rename Notes**: Give your notes custom names for easy reference.  
- **Delete Notes**: Remove unwanted notes, including their summaries and chats.  
- **Export to PDF**: Download your notes, summaries, and chats as a beautifully formatted PDF.

</details>

<details>
<summary><strong>🤖 AI-Powered Chat</strong> (Click to expand)</summary>

- **Ask Questions**: Chat with an AI to get answers based on your PDF content. Perfect for clarifying concepts or digging deeper.  
- **View Chat History**: Revisit past questions and answers for each note.  
- **Daily Limit**: Ask up to 20 questions per day to ensure smooth performance.

</details>

<details>
<summary><strong>🔒 Security & Organization</strong> (Click to expand)</summary>

- **Authentication**: All actions require login to keep your data private.  
- **Note Organization**: See all your notes in a tidy dashboard, sorted by upload date.  
- **Cloud Storage**: Your PDFs are safely stored in the cloud, accessible anytime.

</details>

---

## 🛠 Get Started

Ready to start using Smart Notes? Follow these steps to set it up on your computer.

### Prerequisites
You’ll need:
- **Python** (version 3.8 or higher)  
- A **code editor** (e.g., VS Code)  
- A **web browser** (e.g., Chrome, Firefox)  
- Accounts for these services:  
  - [Supabase](https://supabase.com) (for storing notes and user data)  
  - [Cloudinary](https://cloudinary.com) (for PDF storage)  
  - [Google API](https://cloud.google.com) (for AI features)  
  - [Pinecone](https://www.pinecone.io) (for smart search)  
  - [Hugging Face](https://huggingface.co) (for text processing)

### Installation Steps
1. **Download the Code**:
   - Clone the repository: `git clone https://github.com/krish1440/Smart-Notes`.
   - Navigate to the project folder: `cd smart-notes`.

2. **Set Up a Virtual Environment**:
   - Create: `python -m venv venv`
   - Activate:
     - Windows: `venv\Scripts\activate`
     - Mac/Linux: `source venv/bin/activate`

3. **Install Dependencies**:
   - Run: `pip install -r requirements.txt`
   - This installs libraries like Flask, PyMuPDF, ReportLab, and more.

4. **Configure Environment Variables**:
   - Create a `.env` file in the project folder.
   - Add these settings (replace with your actual keys):
     ```plaintext
     SUPABASE_URL=your_supabase_url
     SUPABASE_KEY=your_supabase_key
     GOOGLE_API_KEY=your_google_api_key
     CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
     CLOUDINARY_API_KEY=your_cloudinary_api_key
     CLOUDINARY_API_SECRET=your_cloudinary_api_secret
     JWT_SECRET=your_jwt_secret
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_INDEX_NAME=your_pinecone_index_name
     HUGGINGFACE_API_KEY=your_huggingface_api_key
5.  **Run the App**:
    - Start the server: python app.py
    ```plaintext
    Open http://localhost:5000 in your browser.
