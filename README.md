# Medical Chatbot – RAG Based Generative AI Project

A domain-specific **Medical Chatbot** built using **Flask, LangChain, Pinecone, Hugging Face embeddings, and Groq/OpenAI-compatible LLM APIs**.  
The chatbot answers user questions by retrieving relevant information from medical PDF documents and generating responses based only on the retrieved context.

---

## Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot designed for the medical domain.  
Instead of answering from general model knowledge alone, the chatbot first searches a medical knowledge base created from PDF documents, retrieves the most relevant chunks, and then uses a Large Language Model (LLM) to generate a grounded answer.

This makes the chatbot more reliable for document-based question answering and reduces hallucinations.

---

## Problem Statement

Traditional chatbots often:
- generate generic answers
- hallucinate information
- cannot answer based on custom documents
- do not provide document-grounded responses

This project solves that by:
- extracting information from medical PDF books/documents
- splitting text into chunks
- converting chunks into vector embeddings
- storing them in Pinecone
- retrieving the most relevant chunks when a user asks a question
- generating the final answer using an LLM

---

## Features

- Upload and process medical PDF documents
- Extract text from PDFs
- Split text into meaningful chunks
- Generate embeddings using Hugging Face
- Store embeddings in Pinecone vector database
- Retrieve relevant chunks using semantic similarity
- Answer user questions using a RAG pipeline
- Web-based chatbot interface using Flask
- Modular code structure for easy maintenance and extension

---

## Tech Stack

### Backend
- **Python**
- **Flask**

### RAG / LLM Framework
- **LangChain**
- **LangChain Classic**
- **LangChain Pinecone**
- **LangChain Hugging Face**

### Embedding Model
- **sentence-transformers/all-MiniLM-L6-v2**

### Vector Database
- **Pinecone**

### LLM
- **Groq API** using OpenAI-compatible endpoint  
  *(can also be switched to OpenAI if required)*

### Document Processing
- **PyPDF**
- **PyPDFLoader**
- **DirectoryLoader**

### Environment Management
- **python-dotenv**
- **conda / venv**

---

## How the Project Works

The project follows the **RAG pipeline**:

### 1. Document Loading
Medical PDF files are loaded from the `Data/` folder using LangChain document loaders.

### 2. Text Cleaning and Preprocessing
The extracted text is filtered and cleaned to keep only useful content.

### 3. Text Chunking
The text is split into smaller chunks using `RecursiveCharacterTextSplitter` with:
- `chunk_size = 500`
- `chunk_overlap = 20`

### 4. Embedding Generation
Each chunk is converted into vector embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

### 5. Vector Storage
The embeddings are stored in a Pinecone index for semantic retrieval.

### 6. Retrieval
When the user asks a question:
- the question is embedded
- the most relevant chunks are retrieved from Pinecone

### 7. Answer Generation
The retrieved context is passed to the LLM with a system prompt.  
The model generates an answer only from the retrieved context.

---

## Project Structure

```bash
Medical-ChatBot-Gen-AI/
│
├── Data/                     # Medical PDF files
├── research/                 # Notebook experiments / trials
│   └── trials.ipynb
│
├── src/
│   ├── __init__.py
│   ├── helper.py             # Helper functions for loading, chunking, embeddings
│   └── prompt.py             # System prompt
│
├── templates/
│   └── chat.html             # Frontend chat page
│
├── app.py                    # Flask application
├── setup.py
├── requirements.txt
├── .env                      # API keys
└── README.md
