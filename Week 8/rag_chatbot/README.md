
# RAG Q&A Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses document retrieval and generative AI to answer questions based on your own CSV data. It uses Hugging Face models and runs locally on Windows.

---

## Features
- Loads and chunks your CSV data for context retrieval
- Uses MiniLM for semantic search (retrieval)
- Uses DistilGPT2 for answer generation
- Simple command-line interface (CLI)

---

## Prerequisites
- Windows OS
- Python 3.12 (or compatible)
- Internet connection (for first-time model downloads)

---

## Setup Instructions

### 1. Clone or Download the Project
Place all files in a folder, e.g. `C:\Users\morey\Downloads\Week 8\rag_chatbot`.

### 2. Place Your Data
Ensure your CSV file is named `Training Dataset.csv` and is located at:
```
C:\Users\morey\Downloads\Week 8\Training Dataset.csv
```

### 3. Install Python Packages
Open PowerShell in the project folder and run:
```
pip install -r requirements.txt
```
If you see errors about missing or incompatible packages, run:
```
pip install torch torchvision transformers sentence-transformers pandas scikit-learn
```

---

## How to Run the Chatbot

Open PowerShell in `C:\Users\morey\Downloads\Week 8` and run:
```
C:/Users/morey/AppData/Local/Programs/Python/Python312/python.exe rag_chatbot/main.py
```

---

## Usage

1. After running, you will see:
   ```
   RAG Q&A Chatbot Ready! Type your question (or "exit" to quit)
   ```
2. Type your question and press Enter.
3. The bot will answer using your CSV data and generative AI.
4. Type `exit` to quit.

---

## File Overview
- `main.py` — Main chatbot logic and CLI
- `retriever.py` — Loads CSV, chunks text, retrieves relevant context
- `requirements.txt` — List of required Python packages
- `README.md` — This guide

---

## Troubleshooting

- **FileNotFoundError**: Make sure your CSV path is correct and matches the code.
- **ModuleNotFoundError**: Install missing packages using pip as shown above.
- **Model Download Issues**: Ensure you have internet for the first run; models are cached after.
- **Memory/Data Usage**: First run may use 300–500 MB for model downloads; later runs use almost no data.

---

## Credits
Built with Hugging Face Transformers and Sentence Transformers.
