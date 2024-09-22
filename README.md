llmops mini capstone project on RAG
# *Mini-capstone-project*

This project demonstrates a **PDF-based Question Answering (QA) system** built with **Chainlit**, using a combination of **FAISS for similarity search**, **Sentence-Transformer for embedding generation**, and **Hugging Face's summarization pipeline** to deliver relevant answers from PDF documents.

## **Features**

- **PDF Parsing:** Extracts text from a user-uploaded PDF file.
- **Text Chunking:** Splits the extracted text into smaller chunks to improve processing efficiency.
- **Sentence Embeddings:** Converts text chunks into sentence embeddings using a pre-trained model (`all-MiniLM-L6-v2`).
- **FAISS Indexing:** Utilizes **FAISS** to create an index and perform fast similarity searches.
- **Text Summarization:** Uses the **BART summarization model** from Hugging Face to generate concise answers from the most relevant text chunk.
- **Interactive UI with Chainlit:** Provides an interactive UI where users can upload a PDF, ask questions, and receive answers directly.

## **Tech Stack**

- **Python latest**
- **Chainlit**: For building a conversational interface.
- **FAISS**: For fast similarity search across document embeddings.
- **PyPDF2**: To extract text from PDFs.
- **Sentence-Transformers**: For sentence embedding generation.
- **Hugging Face Transformers**: For summarizing selected text chunks.

## **Installation**

1. **Install dependencies:**

    ```bash
    pip install all library mentioned in Dependencies below
    ```

2. **Run the application:**

    ```bash
    chainlit run main.py -w
    ```

    This will start the Chainlit application, opening it in your default web browser.

## **How to use it**

1. **Upload a PDF file** using the Chainlit interface.
2. **Ask a question** about the content of the PDF.
3. The system will:
   - Parse the PDF,
   - Split the text into chunks,
   - Generate embeddings for the chunks,
   - Perform a similarity search to find the most relevant chunk based on the question,
   - Summarize the selected text chunk,
   - Return the summary as the answer to your question.

4. To stop the process, type `STOP` in the question prompt.

## **File Structure**
├── project/
│   ├── local_cola/            #pdf local repo             
├── main.py                    #Main code to run                        
└── README.md                  #Ream it             


## **Dependencies**

- **Chainlit**: `pip install chainlit`
- **FAISS**: `pip install faiss-cpu`
- **PyPDF2**: `pip install PyPDF2`
- **Sentence-Transformers**: `pip install sentence-transformers`
- **Transformers**: `pip install transformers`
- **Numpy**: `pip install numpy`


