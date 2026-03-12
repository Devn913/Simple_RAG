import os
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .forms import PDFUploadForm
from .models import PDFDocument
from .utils import process_pdf, get_answer
from django.conf import settings

def index(request):
    documents = PDFDocument.objects.all().order_by('-uploaded_at')
    form = PDFUploadForm()
    return render(request, 'chat/index.html', {'documents': documents, 'form': form})

import logging

logger = logging.getLogger(__name__)

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_doc = form.save()
            
            # Ensure vector_stores directory exists
            vector_store_root = os.path.join(settings.BASE_DIR, 'vector_stores')
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
        
        if not query or not doc_id:
            return JsonResponse({'error': 'Missing query or document selection'}, status=400)
            
        try:
            pdf_doc = PDFDocument.objects.get(id=doc_id)
            if not pdf_doc.vector_store_path or not os.path.exists(pdf_doc.vector_store_path):
                return JsonResponse({'error': 'Vector store not found for this document'}, status=404)
            
            answer = get_answer(query, pdf_doc.vector_store_path)
            return JsonResponse({'answer': answer})
        except PDFDocument.DoesNotExist:
            return JsonResponse({'error': 'Document not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)
