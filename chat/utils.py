import os
import google.generativeai as genai
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import logging
from google import genai
from django.conf import settings
...
def list_gemini_models():
    """
    Lists available Gemini models that support content generation.
    """
    api_key = getattr(settings, 'GOOGLE_API_KEY', None)
    if not api_key:
        return [
            {'name': 'models/gemini-1.5-flash', 'display_name': 'Gemini 1.5 Flash (Default)'},
            {'name': 'models/gemini-1.5-pro', 'display_name': 'Gemini 1.5 Pro'},
        ]
    
    try:
        # We use the older SDK for listing as it's already configured in other parts of the app
        # and has better compatibility with the current environment
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and m.name.startswith('models/gemini'):
                # Clean up display name if needed
                display_name = m.display_name
                if not display_name or display_name == m.name:
                    display_name = m.name.replace('models/', '').replace('-', ' ').title()
                
                models.append({
                    'name': m.name,
                    'display_name': display_name
                })
        
        # Sort models to show latest/popular ones first (simple heuristic)
        models.sort(key=lambda x: x['name'], reverse=True)
        
        return models if models else [
            {'name': 'models/gemini-1.5-flash', 'display_name': 'Gemini 1.5 Flash'},
            {'name': 'models/gemini-1.5-pro', 'display_name': 'Gemini 1.5 Pro'},
        ]
    except Exception as e:
        logger.error(f"Error listing Gemini models: {str(e)}")
        return [
            {'name': 'models/gemini-1.5-flash', 'display_name': 'Gemini 1.5 Flash'},
            {'name': 'models/gemini-1.5-pro', 'display_name': 'Gemini 1.5 Pro'},
        ]

def process_pdf(file_path, vector_store_path):
    """
    Loads a PDF, splits it into chunks, creates embeddings, and saves to a FAISS vector store.
    """
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Create and save FAISS vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(vector_store_path)
    
    return vectorstore

def get_answer(query, vector_store_path, model_name="models/gemini-1.5-flash"):
    """
    Loads the FAISS vector store and uses Gemini to answer the query based on the context.
    """
    print(f"--- Asking question: {query} using model: {model_name} ---")
    # 1. Load embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded.")

    # 2. Initialize Gemini LLM with the selected model
    # Use a fallback if model_name is invalid or not provided
    if not model_name:
        model_name = "models/gemini-1.5-flash"
    
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    print(f"LLM initialized ({model_name}).")

    # 3. Define the prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    print("Prompt defined.")

    # 4. Create retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    print("Chain created.")

    # 5. Get response
    try:
        print("Invoking chain...")
        response = rag_chain.invoke({"input": query})
        print("Chain invoked successfully.")
        return response["answer"]
    except Exception as e:
        print(f"Chain error: {str(e)}")
        raise e
