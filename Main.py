import chainlit as cl
import PyPDF2
import shutil
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Specify the directory where you want to save the uploaded PDF
SAVE_DIRECTORY = ".\project\local_cola"
# Create the directory if it doesn't exist
os.makedirs(SAVE_DIRECTORY, exist_ok=True)
welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF file
2. Ask a question about the file
"""
# Global variables to store embeddings and document chunks

embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model

# Function to parse PDF and split text into smaller chunks
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Split the text into chunks of size 1000 characters (approx.)
    overlap = 900  # Number of overlapping characters
    chunk_size = 1200  # Size of each chunk
    if len(text) < chunk_size:
        chunks.append(text)  # If the text is shorter than the chunk size, just add it as a chunk.
    else:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap)]


    return chunks

# Function to generate embeddings from document chunks
def generate_embeddings(chunks):
    global embedder
    embeddings = embedder.encode(chunks)
    return embeddings

# Function to create FAISS index and add embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Function to perform similarity search using FAISS
def perform_similarity_search(index, user_question):
    query_embedding = embedder.encode([user_question])
    query_embedding = np.array(query_embedding).reshape(1, -1)
    k = 2  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k=k)
    return indices

# Function to summarize the selected chunk of text
def summarize_text(chunk):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(chunk, max_length=1000, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Main function to handle the flow in Chainlit
# start
@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message, accept=["application/pdf"]
        ).send()

    pdf_file = files[0]

    # # Read the PDF file
    text = ""
    with open(pdf_file.path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""

    # Save the uploaded PDF to the specified location
    destination_path = os.path.join(SAVE_DIRECTORY, pdf_file.name)
    shutil.copy(pdf_file.path, destination_path)

    # Let the user know that the system is ready
    await cl.Message(
        content=f"`{pdf_file.name}` uploaded and saved to `{destination_path}`, it contains {len(text)} characters!"
    ).send()
    while True:
        # embeddings = None
        # chunks = None
        # user_question = await cl.AskUserMessage(content="Ask a question").send()
        user_question = await cl.AskUserMessage(content="Ask a question (type 'STOP' to exit)").send()

        # Display the user's question in the UI
        print(user_question['output'],type(user_question['output']))
        if user_question['output'] in ['STOP', 'Stop', 'stop', '']:
            await cl.Message(content="\n*It’s my pleasure to serve,\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tplease come back with another task").send()
            print()
            break
        await cl.Message(content=f"*Question:*\n'{user_question['output']}'").send()
        
        # # Check for a valid user question
        # if user_question is None or user_question.strip() == "":
        #     await cl.Message(content="Please enter a valid question.").send()
        #     continue  # Re-prompt for a question if input is invalid
        # user_question = await cl.AskUserMessage(content="Ask a question", timeout=10).send()   
        # Step 2: Parse the uploaded PDF and extract text in chunks
    
        chunks = extract_text_from_pdf(destination_path)
    
        # Step 3: Generate embeddings for the document chunks
        embeddings = generate_embeddings(chunks)
    
        # Step 4: Create FAISS index with generated embeddings
        index = create_faiss_index(embeddings)
    
        # Step 6: Perform similarity search using FAISS to find relevant chunks
        indices = perform_similarity_search(index, user_question['output'])
    
        # Step 7: Retrieve the top chunk based on the search results
        top_chunk_index = indices[0][0]
        top_chunk_text = chunks[top_chunk_index]

        # Step 8: Summarize the top chunk using the summarization model
        summary = summarize_text(top_chunk_text)

        # Step 9: Return the summarized answer to the user
        await cl.Message(content=f"**************Answer*************\n{summary}").send()
        # break