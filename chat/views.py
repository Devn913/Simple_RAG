import os
import shutil
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.conf import settings
from .forms import PDFUploadForm
from .models import PDFDocument
from .utils import process_pdf, get_answer, list_gemini_models, validate_key

logger = logging.getLogger(__name__)

def index(request):
    documents = PDFDocument.objects.all().order_by('-uploaded_at')
    form = PDFUploadForm()
    
    # Check if we should use server keys (admin logged in) or user keys
    use_server_keys = request.user.is_authenticated
    
    # If not admin, we might have keys in the session
    gemini_key = request.session.get('gemini_key')
    openai_key = request.session.get('openai_key')
    
    available_models = list_gemini_models(api_key=None if use_server_keys else gemini_key)
    
    # Add common OpenAI models if key is provided
    if openai_key:
        available_models.extend([
            {'name': 'gpt-4o', 'display_name': 'GPT-4o'},
            {'name': 'gpt-4-turbo', 'display_name': 'GPT-4 Turbo'},
            {'name': 'gpt-3.5-turbo', 'display_name': 'GPT-3.5 Turbo'},
        ])

    return render(request, 'chat/index.html', {
        'documents': documents, 
        'form': form,
        'available_models': available_models,
        'is_admin': request.user.is_authenticated,
        'has_gemini_key': bool(gemini_key),
        'has_openai_key': bool(openai_key)
    })

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return render(request, 'chat/login.html', {'error': 'Invalid credentials'})
    return render(request, 'chat/login.html')

def logout_view(request):
    logout(request)
    return redirect('index')

@require_POST
def validate_api_key(request):
    key_type = request.POST.get('key_type')
    key = request.POST.get('key')
    
    if not key:
        return JsonResponse({'success': False, 'error': 'Key is required'})
    
    success, message = validate_key(key_type, key)
    if success:
        # Save to session
        request.session[f'{key_type}_key'] = key
        return JsonResponse({'success': True, 'message': message})
    else:
        return JsonResponse({'success': False, 'error': message})

@require_POST
def clear_keys(request):
    request.session.pop('gemini_key', None)
    request.session.pop('openai_key', None)
    return JsonResponse({'success': True})

def upload_pdf(request):
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Check if user has permission (admin or has provided keys)
            gemini_key = request.session.get('gemini_key')
            openai_key = request.session.get('openai_key')
            
            if not request.user.is_authenticated and not (gemini_key or openai_key):
                return redirect('index')

            pdf_doc = form.save()
            
            vector_store_root = os.path.join(getattr(settings, 'DATA_DIR', settings.BASE_DIR), 'vector_stores')
            if not os.path.exists(vector_store_root):
                os.makedirs(vector_store_root)
            
            folder_name = f"doc_{pdf_doc.id}"
            vector_store_path = os.path.join(vector_store_root, folder_name)
            
            pdf_doc.vector_store_path = vector_store_path
            pdf_doc.save()
            
            try:
                # Use server keys if admin, otherwise session keys
                g_key = None if request.user.is_authenticated else gemini_key
                o_key = None if request.user.is_authenticated else openai_key
                
                process_pdf(pdf_doc.file.path, vector_store_path, gemini_key=g_key, openai_key=o_key)
                pdf_doc.status = 'COMPLETED'
                pdf_doc.save()
            except Exception as e:
                pdf_doc.status = 'FAILED'
                pdf_doc.save()
                logger.error(f"Error processing PDF: {str(e)}")
            
            return redirect('index')
    return redirect('index')

def ask_question(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        doc_id = request.POST.get('doc_id')
        model_name = request.POST.get('model_name')
        
        gemini_key = request.session.get('gemini_key')
        openai_key = request.session.get('openai_key')
        
        if not request.user.is_authenticated and not (gemini_key or openai_key):
            return JsonResponse({'error': 'Please provide an API key first'}, status=401)

        if not query or not doc_id:
            return JsonResponse({'error': 'Missing query or document selection'}, status=400)
            
        try:
            pdf_doc = PDFDocument.objects.get(id=doc_id)
            if not pdf_doc.vector_store_path or not os.path.exists(pdf_doc.vector_store_path):
                return JsonResponse({'error': 'Vector store not found'}, status=404)
            
            g_key = None if request.user.is_authenticated else gemini_key
            o_key = None if request.user.is_authenticated else openai_key
            
            answer = get_answer(query, pdf_doc.vector_store_path, model_name=model_name, gemini_key=g_key, openai_key=o_key)
            return JsonResponse({'answer': answer})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid method'}, status=405)

@require_POST
def delete_document(request, doc_id):
    # Only allow admin to delete? Or anyone? User asked for admin login to use personal keys.
    # For now, let's keep it accessible but maybe restrict to admin later if needed.
    try:
        pdf_doc = get_object_or_404(PDFDocument, id=doc_id)
        if pdf_doc.vector_store_path and os.path.exists(pdf_doc.vector_store_path):
            shutil.rmtree(pdf_doc.vector_store_path)
        if pdf_doc.file and os.path.exists(pdf_doc.file.path):
            os.remove(pdf_doc.file.path)
        pdf_doc.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
