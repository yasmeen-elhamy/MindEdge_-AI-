# 🧠 MindEdge – Manager AI 
### Intelligent Learning Assistant (AI-Powered Backend)

MindEdge is an AI-powered educational backend system designed to transform uploaded learning materials into **structured knowledge**, **interactive quizzes**, and **personalized learning experiences** using OCR, Machine Learning, and Large Language Models (LLMs).

This repository contains the **backend manager layer** responsible for data handling, AI orchestration, user management, and learning intelligence services.

---

## 🚀 Key Features

### 👤 User Management
- Secure user authentication & authorization
- Role-based access control (Student / Admin)
- Persistent learning history and progress tracking

### 📄 Document Processing
- Upload scanned images and PDFs
- OCR text extraction and cleaning
- Document lifecycle management

### 🧠 AI & Intelligence Layer
- AI-powered notes generation
- Context-aware Q&A assistant
- Retrieval-Augmented Generation (RAG)
- Model-agnostic LLM integration via API

### 🧪 Assessment & Evaluation
- Automatic quiz generation
- Multiple question types
- Quiz attempts & scoring
- Performance analytics

### 📈 Personalized Learning
- ML-based recommendation engine
- Adaptive study plans
- Learning pattern analysis

```
#🏗️ System Architecture (High Level)

Client (Web / Mobile)
│
└── API Gateway
    │
    └── MindEdge Backend
        ├── Auth & User Service
        ├── Document & OCR Service
        ├── AI & RAG Engine
        ├── Quiz & Assessment Engine
        ├── Recommendation Engine
        └── Database Layer
```
## 🧩 Database Design

- Relational database (MySQL)
- Fully normalized (3NF)
- Designed using EER modeling
- Supports inheritance and weak entities

**EER Highlights:**
- User → Student / Admin (Inheritance)
- User_Quiz as weak entity
- AI interaction and recommendation tracking

> 📌 See EER Diagram in project documentation.

---

## 🛠️ Tech Stack

### Backend
-Python / FastAPI
- Uvicorn
- RESTful APIs
- JWT Authentication

### Database
- MySQL
- MySQL Workbench (EER Modeling)

### AI / ML
- OCR Engine (Tesseract or equivalent)
- LLM via external API
- Retrieval-Augmented Generation (RAG)
- ML-based recommendation logic

## 📁 Project Structure
```
MindEdge-Manager-Backend/
│
├── src/
│   ├── controllers/        # Request handlers
│   ├── routes/             # API routes
│   ├── services/           # Business logic & AI orchestration
│   ├── models/             # Database models
│   ├── middlewares/        # Authentication & validation
│   └── utils/              # Helper utilities
│
├── build/                  # Compiled production build
├── .env.example            # Environment variables template
├── package.json
├── README.md
└── docs/                   # Diagrams & documentation
```
## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```env
# Server
PORT=3000
NODE_ENV=development

# Database
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=mindedge_db

# Authentication
JWT_SECRET=your_secret_key

# AI Service
AI_HOST=localhost
AI_PORT=8000
