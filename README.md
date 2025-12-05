# Intelligent Test Case Search Engine

This project provides a **web-based platform** for uploading software test case files (`CSV` / `XLSX`), enriching them using the **Google Gemini API**, and storing them in **MongoDB with vector embeddings** to enable **fast and accurate semantic search**.

The system combines LLM-based data enrichment with modern vector databases to deliver **intelligent retrieval of test cases** using natural language queries.

---

---

## Features

- **File Upload**
  - Supports both `.csv` and `.xlsx` formats.

- **AI Enrichment**
  - Automatically generates summaries and keywords for each test case using Google Gemini.

- **Vector Search**
  - Utilizes **Sentence-Transformers** and **MongoDB Atlas Vector Search** to locate semantically similar test cases.

- **Web Interface**
  - Clean and simple UI built using:
    - FastAPI
    - HTML
    - Tailwind CSS

- **REST API**
  - Endpoints for:
    - Uploading test cases
    - Searching existing records
    - Updating test cases
    - Deleting records

---

---

## Project Structure

```

.
├── main.py          # FastAPI backend application
├── index.html      # Frontend UI
├── requirements.txt
└── README.md

````

---

---

## Setup and Installation

---

### Step 1 — Prerequisites

Ensure the following are available:

- **Python 3.8+**
- **MongoDB Atlas**
  - Create a free cluster and obtain your connection string.
- **Google Gemini API Key**
  - Visit Google AI Studio
  - Click **Get API key**
  - Create and save your API key

---

---

### Step 2 — Create Virtual Environment

Create and activate a Python virtual environment.

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
````

#### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

---

## Configuration

Open `main.py` and update the credentials:

```python
# MongoDB connection string
MONGO_CONNECTION_STRING = "YOUR_MONGODB_ATLAS_URI"

# Google Gemini API key
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

---

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

**Command Breakdown**

| Component  | Description                         |
| ---------- | ----------------------------------- |
| `uvicorn`  | ASGI server                         |
| `main:app` | Loads `app` instance from `main.py` |
| `--reload` | Enables auto-reload on code changes |

---

---

## Using the Application

Open your browser:

```
http://127.0.0.1:8000
```

---

### Upload Flow

1. Click **Choose File**
2. Select a `.csv` or `.xlsx` file
3. Click **Upload and Process**

---

### Search Flow

1. Enter your query in the search box
2. Click **Search**
3. View the top matching test cases ranked by semantic similarity

---

---

## How It Works: Multi-Level Indexing

This implementation uses MongoDB **Vector Search** for semantic retrieval.

For large datasets, performance can be significantly enhanced using a **multi-level indexing strategy**:

---

### Two-Step Retrieval Concept

```text
Query
   ↓
LLM Feature Classification
   ↓
Metadata Filtering (Ex: Feature = "Login Page")
   ↓
MongoDB Vector Search (within filtered subset)
   ↓
Final semantic ranking
```

---

### Benefits

* Smaller search space
* Faster response times
* Improved relevance accuracy
* Lower compute cost

---

---

## MongoDB Vector Index Configuration

Create a vector index on the field **`main_vector`**:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "main_vector",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

---

### Index Name

```
vector_index
```

---

---

## Summary

This project demonstrates a practical implementation of **LLM-powered semantic search** for software test cases, combining:

* Gemini-based AI enrichment
* SentenceTransformer embeddings
* MongoDB vector indexing
* A simple web UI and REST interface

