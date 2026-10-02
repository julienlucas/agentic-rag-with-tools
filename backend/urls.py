from .views import index, upload_file, process_question, load_file, list_documents, delete_document, delete_space
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = [
    path('', index),
    path('a-propos', index),
    path('a-propos/', index),
    path('api/load-file', load_file),
    path('api/upload-file', upload_file),
    path('api/process-question', process_question),
    path('api/documents', list_documents),
    path('api/delete-document', delete_document),
    path('api/delete-space', delete_space),
]

if settings.DEBUG:
    for static_dir in settings.STATICFILES_DIRS:
        urlpatterns += static(settings.STATIC_URL, document_root=static_dir)

    assets_dir = settings.BASE_DIR / "frontend" / "dist" / "assets"
    if assets_dir.exists():
        urlpatterns += [
            path('assets/<path:path>', serve, {'document_root': str(assets_dir)}),
        ]
