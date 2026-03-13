import os
import shutil
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .forms import PDFUploadForm
from .models import PDFDocument
from .utils import process_pdf, get_answer, list_gemini_models
from django.conf import settings

def index(request):
    documents = PDFDocument.objects.all().order_by('-uploaded_at')
    form = PDFUploadForm()
    # Fetch available models from Gemini
    available_models = list_gemini_models()
    return render(request, 'chat/index.html', {
        'documents': documents, 
        'form': form,
        'available_models': available_models
    })

import logging

logger = logging.getLogger(__name__)

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_doc = form.save()
            
            # Ensure vector_stores directory exists
            vector_store_root = os.path.join(getattr(settings, 'DATA_DIR', settings.BASE_DIR), 'vector_stores')
            if not os.path.exists(vector_store_root):
                os.makedirs(vector_store_root)
            
            # Use a more robust path for the vector store
            folder_name = f"doc_{pdf_doc.id}"
            vector_store_path = os.path.join(vector_store_root, folder_name)
            
            # Update the path in the model
            pdf_doc.vector_store_path = vector_store_path
            pdf_doc.save()
            
            # Process the PDF (extract, chunk, embed)
            try:
                process_pdf(pdf_doc.file.path, vector_store_path)
                pdf_doc.status = 'COMPLETED'
                pdf_doc.save()
                logger.info(f"Successfully processed PDF: {pdf_doc.title}")
            except Exception as e:
                pdf_doc.status = 'FAILED'
                pdf_doc.save()
                logger.error(f"Error processing PDF '{pdf_doc.title}': {str(e)}")
            
            return redirect('index')
    return redirect('index')

def ask_question(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        doc_id = request.POST.get('doc_id')
        model_name = request.POST.get('model_name')
        
        if not query or not doc_id:
            return JsonResponse({'error': 'Missing query or document selection'}, status=400)
            
        try:
            pdf_doc = PDFDocument.objects.get(id=doc_id)
            if not pdf_doc.vector_store_path or not os.path.exists(pdf_doc.vector_store_path):
                return JsonResponse({'error': 'Vector store not found for this document'}, status=404)
            
            # Pass the selected model_name to get_answer
            answer = get_answer(query, pdf_doc.vector_store_path, model_name=model_name)
            return JsonResponse({'answer': answer})
        except PDFDocument.DoesNotExist:
            return JsonResponse({'error': 'Document not found'}, status=404)
        except Exception as e:
            logger.error(f"Error in ask_question: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def delete_document(request, doc_id):
    if request.method == 'POST':
        try:
            pdf_doc = get_object_or_404(PDFDocument, id=doc_id)
            
            # 1. Delete vector store folder if it exists
            if pdf_doc.vector_store_path and os.path.exists(pdf_doc.vector_store_path):
                try:
                    shutil.rmtree(pdf_doc.vector_store_path)
                    logger.info(f"Deleted vector store for document ID {doc_id}")
                except Exception as e:
                    logger.error(f"Error deleting vector store: {str(e)}")
            
            # 2. Delete the PDF file itself
            if pdf_doc.file and os.path.exists(pdf_doc.file.path):
                try:
                    os.remove(pdf_doc.file.path)
                    logger.info(f"Deleted PDF file for document ID {doc_id}")
                except Exception as e:
                    logger.error(f"Error deleting PDF file: {str(e)}")
            
            # 3. Delete from DB
            pdf_doc.delete()
            return JsonResponse({'success': True, 'message': 'Document and associated data deleted successfully'})
            
        except Exception as e:
            logger.error(f"Error in delete_document view: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)
