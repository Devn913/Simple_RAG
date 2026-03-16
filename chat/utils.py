import os
import google.generativeai as genai
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

def list_gemini_models(api_key=None):
    """
    Lists available Gemini models that support content generation.
    """
    api_key = api_key or getattr(settings, 'GOOGLE_API_KEY', None)
    if not api_key:
        return [
            {'name': 'models/gemini-1.5-flash', 'display_name': 'Gemini 1.5 Flash (Default)'},
            {'name': 'models/gemini-1.5-pro', 'display_name': 'Gemini 1.5 Pro'},
        ]
    
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and m.name.startswith('models/gemini'):
                display_name = m.display_name
                if not display_name or display_name == m.name:
                    display_name = m.name.replace('models/', '').replace('-', ' ').title()
                
                models.append({
                    'name': m.name,
                    'display_name': display_name
                })
        
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

def get_embeddings(gemini_key=None, openai_key=None):
    """
    Returns the appropriate embeddings object based on available keys.
    Prefers OpenAI if key is provided, otherwise Gemini.
    """
    if openai_key:
        return OpenAIEmbeddings(openai_api_key=openai_key)
    
    api_key = gemini_key or getattr(settings, 'GOOGLE_API_KEY', None)
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

def process_pdf(file_path, vector_store_path, gemini_key=None, openai_key=None):
    """
    Loads a PDF, splits it into chunks, creates embeddings, and saves to a FAISS vector store.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)

    embeddings = get_embeddings(gemini_key=gemini_key, openai_key=openai_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(vector_store_path)
    
    return vectorstore

def get_answer(query, vector_store_path, model_name="models/gemini-1.5-flash", gemini_key=None, openai_key=None):
    """
    Loads the FAISS vector store and uses the selected model to answer the query.
    """
    embeddings = get_embeddings(gemini_key=gemini_key, openai_key=openai_key)
    vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    if openai_key and model_name.startswith('gpt'):
        llm = ChatOpenAI(model=model_name, temperature=0.3, openai_api_key=openai_key)
    else:
        api_key = gemini_key or getattr(settings, 'GOOGLE_API_KEY', None)
        # Use fallback if model_name is for OpenAI but we only have Gemini
        if model_name.startswith('gpt'):
            model_name = "models/gemini-1.5-flash"
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3, google_api_key=api_key)

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        logger.error(f"Chain error: {str(e)}")
        raise e

def validate_key(key_type, key):
    """
    Validates the provided API key by making a simple request.
    """
    try:
        if key_type == 'gemini':
            genai.configure(api_key=key)
            genai.list_models()
            return True, "Gemini key is valid."
        elif key_type == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=key)
            client.models.list()
            return True, "OpenAI key is valid."
    except Exception as e:
        return False, str(e)
    return False, "Unknown key type."
