import os
import json
import time
import pymysql
import re
import spacy
import pickle
from django.core.cache import cache
from docx import Document
from django.http import JsonResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from django.conf import settings

nlp = spacy.load("en_core_web_sm")

## Initialization
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

VECTOR_STORE = None
FAISS_INDEX_PATH = "faiss_index.pkl"

MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306)),
}
## Hugging face embedding model to convert text chunks into numerical vectors.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant specialized in answering customer queries.  
    Provide accurate, in-depth, and concise responses strictly based on the given context and Chat History.  

    **Guidelines:**  
    - Your response must be **precise and to the point**.  
    - **Do not** use phrases like "According to the provided context."  
    - **Never** disclose API keys, passwords, or personal user information.  
    - **If the answer is not in the context, respond with:** "I don't have enough information to answer this." 
    Chat History:
    {chat_history}

    <context>
    {context}
    </context>
     question: {question} 
    """
)


## Memory concept :
CHAT_HISTORY_LIMIT = 10
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    output_key="answer",
    k=CHAT_HISTORY_LIMIT,
    return_messages=True,
)


def get_chat_history(memory):
    memory.load_memory_variables({}).get("chat_history", [])


def clear_chat_history():
    """Clear chat memory."""
    memory.clear()
    print("Chat history cleared.")


## Database operations:
def connect_db():
    return pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)


## Storing embedings into DB.
def store_vector_in_db(file_name, chunk_text, embedding):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO vector_store (file_name, chunk_text, embedding) VALUES (%s, %s, %s)"
            cursor.execute(sql, (file_name, chunk_text, json.dumps(embedding)))
        conn.commit()
    except Exception as e:
        print(f"Error storing vector: {e}")
        conn.rollback()
    finally:
        conn.close()


## Loading chunks,and vector Embeddings from DB/
def load_vectors_from_db():
    """Load vectors from MySQL."""
    global VECTOR_STORE
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT chunk_text, embedding FROM vector_store LIMIT 10000")
            data = cursor.fetchall()
        ## DB is Empty
        if not data:
            print("No vector data found in the database.")
            return

        documents = []
        embeddings_list = []

        for d in data:
            try:
                chunk_text = d["chunk_text"]
                embedding = json.loads(d["embedding"])

                if not isinstance(embedding, list):
                    print(f"Invalid embedding format: {embedding}")
                    continue

                documents.append(chunk_text)
                embeddings_list.append(embedding)
            except Exception as e:
                print(f"Error processing embedding: {e}")

        print(f"Loaded {len(embeddings_list)} embeddings successfully.")
        if len(embeddings_list) != len(documents):
            print("Warning: Mismatch between documents and embeddings!")
        VECTOR_STORE = FAISS.from_texts(documents, embeddings)
        save_faiss_index()

    except Exception as e:
        print(f"Error loading vectors: {e}")
    finally:
        conn.close()


def save_faiss_index():
    """Save the FAISS index to a Pickle file."""
    global VECTOR_STORE
    if VECTOR_STORE is not None:
        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump(VECTOR_STORE, f)
        print("FAISS index saved successfully.")


## Index file for optimized data retrievals.
def load_faiss_index():
    """Load the FAISS index from Pickle if available."""
    global VECTOR_STORE
    if os.path.exists(FAISS_INDEX_PATH):
        with open(FAISS_INDEX_PATH, "rb") as f:
            VECTOR_STORE = pickle.load(f)
        print("FAISS index loaded from Pickle.")
        return True
    return False


## This function is used to Retrieve vector embeddings either from index file or MySQL DB.
def load_index_or_db():
    """Ensures FAISS index is loaded, otherwise loads from DB."""
    global VECTOR_STORE
    if not load_faiss_index():
        load_vectors_from_db()
    return VECTOR_STORE is not None


## Extracting information from JSON Data.
def extract_json_text_as_chunks(data):
    """Convert each JSON object into a single chunk."""
    if isinstance(data, list):
        return [json.dumps(item, ensure_ascii=False) for item in data]
    elif isinstance(data, dict):
        return [json.dumps(data, ensure_ascii=False)]
    print("Unexpected JSON format: Root element is not a dict or list.")
    return []


## This function triggers when user uploads new document.
def process_new_document(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        text_entries = []

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            documents = loader.load()
            text_entries = text_splitter.split_documents(documents)
        elif file_path.endswith(".csv"):
            text_entries = [
                doc.page_content for doc in CSVLoader(file_path=file_path).load()
            ]
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            if json_data:
                text_entries = extract_json_text_as_chunks(json_data)
            else:
                print(f"JSON file is empty: {file_path}")
                return
        else:
            print(f"Unsupported file format: {file_path}")
            return

        if not text_entries:
            print(f"No valid text extracted from: {file_path}")
            return

        for text in text_entries:
            embedding = embeddings.embed_query(text)
            store_vector_in_db(os.path.basename(file_path), text, embedding)

        load_vectors_from_db()
        ## Remove file from Media folder.
        os.remove(file_path)
        print(f"File processed and deleted: {file_path}")
        return True
    except Exception as e:
        print(f"Error processing document ({file_path}): {e}")
        return False


def save_unanswered_question(question):
    """Store unanswered questions in a DOCX file inside the media folder."""
    media_path = os.path.join(settings.MEDIA_ROOT, "unanswered_questions.docx")

    if os.path.exists(media_path):
        doc = Document(media_path)
    else:
        doc = Document()
        doc.add_heading("Unanswered Questions", level=1)

    existing_questions = [
        para.text
        for para in doc.paragraphs
        if para.text and para.text.strip() and not para.style.name.startswith("Heading")
    ]

    if question not in existing_questions:
        doc.add_paragraph(f"{len(existing_questions) + 1}. {question}")

    # Save the updated document
    doc.save(media_path)


# Global variable to store retrieval_chain in memory
retrieval_chain_instance = None


def get_retrieval_chain():
    global retrieval_chain_instance

    if retrieval_chain_instance is not None:
        return retrieval_chain_instance

    if not load_index_or_db():
        return None

    retriever = cache.get("retriever")
    if retriever is None:
        retriever = VECTOR_STORE.as_retriever()
        cache.set("retriever", retriever, timeout=3600)

    retrieval_chain_instance = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        return_source_documents=True,
    )
    return retrieval_chain_instance


def store_retrieved_documents(user_query, response, save_results=False):
    """
    Stores the retrieved source documents in a Word file for performance measurement.
    """
    if not save_results:
        return None

    folder_name = "media/retrieved_docs"
    os.makedirs(folder_name, exist_ok=True)

    doc = Document()
    safe_filename = re.sub(r'[\\/*?:"<>|]', "_", user_query)
    doc_filename = os.path.join(folder_name, f"{safe_filename}.docx")
    doc.add_heading(f"Results for: {user_query}", level=1)

    for i, content in enumerate(response.get("source_documents", []), start=1):
        doc.add_heading(f"Document {i}", level=2)
        doc.add_paragraph(content.page_content)

    doc.save(doc_filename)
    return doc_filename


## This is the Index view of Django.
def index(request):
    if request.method == "POST":
        ## Document Upload section.
        if "document" in request.FILES:
            file = request.FILES["document"]
            file_path = default_storage.save(f"media/{file.name}", file)
            if process_new_document(file_path):
                return render(request, "index.html", {"message": "Upload successful!"})
            return render(request, "index.html", {"error": "Upload failed!"})

        elif "query" in request.POST:
            ## First check the cache if chain not found create it.
            retrieval_chain = get_retrieval_chain()
            if retrieval_chain is None:
                return JsonResponse(
                    {
                        "error": "No indexed documents found. Please upload some files first."
                    },
                    status=400,
                )

            user_query = request.POST["query"]

            ## Asking Query.
            start_time = time.process_time()
            response = retrieval_chain.invoke({"question": user_query})
            response_time = time.process_time() - start_time

            ## This is the code to find out if we have recieved a valid answer or we need to save this Question in unanswered questions document.
            answer = response.get("answer", "").strip()
            if not answer or any(
                substring in answer.lower()
                for substring in [
                    "i don't know",
                    "i'm not sure",
                    "i don't have enough information",
                    "no relevant information",
                ]
            ):
                save_unanswered_question(user_query)
            ## Storing the Source Documents for Performance Measure.
            file_name = store_retrieved_documents(user_query, response, False)

            return JsonResponse(
                {
                    "answer": answer,
                    # "document_link": f"/media/retrieved_docs/{os.path.basename(file_name)}",
                    "response_time": response_time,
                }
            )

    return render(request, "index.html")
