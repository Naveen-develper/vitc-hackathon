from pathlib import Path
from models.mri_model import load_mri_model, predict_mri

# Resolve backend root (one level up from services/)
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Cache
_cache_mri = {}

def init_mri_models(device='cpu'):
    # Load only 3D by default; uncomment 2D if needed
    _cache_mri['3d'] = load_mri_model('3d', device)

try:
    init_mri_models()
except Exception as e:
    # Print a clear warning but allow app to continue
    print(f"Warning: Could not initialize MRI models: {e}")
    _cache_mri = {}  # Ensure cache is empty if initialization fails

def process_mri(path: str, mode: str = '3d', device: str = 'cpu', top_k: int = 2):
    if mode not in _cache_mri or not _cache_mri[mode]:
        raise RuntimeError(f"MRI model for mode '{mode}' not initialized. Please check model files.")
    return predict_mri(_cache_mri[mode], path, mode, device, top_k)

def is_supported_mri_file(filename: str, mode: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in (['.png', '.jpg', '.jpeg'] if mode == '2d' else ['.nii', '.nii.gz', '.dcm'])
