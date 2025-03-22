# Models module 
from .deepfake_model import DeepfakeModel
from .document_forgery_model import DocumentForgeryModel
from .signature_forgery_model import SignatureForgeryModel

__all__ = ["DeepfakeModel", "DocumentForgeryModel", "SignatureForgeryModel"] 