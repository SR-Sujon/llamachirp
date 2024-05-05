# PDF Reading with RAG Implementation and Ollama Models
This repository contains a Python project that leverages RAG (Retrieval-Augmented Generation) implementation along with Ollama models for reading PDF documents and enabling conversational interactions. The system is designed to analyze, correlate, and extract relevant information from the provided PDF context while facilitating user inquiries through conversational interfaces.

## Features
- PDF Processing: The system ingests PDF documents uploaded by the user and extracts text content for further analysis.
- Chunking: Utilizes a Recursive Character Text Splitter to segment the PDF text into manageable chunks.
- Vectorization: Implements OllamaEmbeddings for embedding text chunks into vector representations using the Chroma vector store.
- Conversational Interaction: Engages users in conversations facilitated by ConversationalRetrievalChain, employing Ollama models for generating responses.
- Model Selection: Users can select from various Ollama models to customize the conversational experience.
- Source Tracking: Maintains metadata about the source documents to provide context and source references for generated responses.

## Installation
To install the necessary dependencies, run from terminal:
```pip install -r requirements.txt` ```
To install ollama models, run:
``` ollama pull llama3``` 
``` ollama pull llama2```
``` ollama pull [other model names]```

## Deploy
Run this script in the terminal:
```chainlit run app.py```

## Usage
- Run the Python scripts after installing the dependencies.
- Upload a PDF file when prompted to initiate the conversation and wait for the system to get ready. [Time depends on the PC configuration]
- Select an Ollama model from the provided options.
- Engage in conversation by asking questions or providing prompts.
- Receive precise and accurate responses generated by the system.

## Requirements
- Python 3.6+
- PyPDF2
- langchain_community
- chainlit
- wandb (optional, for tracing)

## Contributing
Contributions are welcome! Please feel free to submit pull requests or raise issues for any improvements or suggestions.

## License
This project is licensed under the MIT License.