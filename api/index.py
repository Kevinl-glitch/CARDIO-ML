import sys
import os

# Cross-platform compatibility for Cython extension modules in scikit-learn
try:
    import sklearn._loss._loss as _loss_cython
    sys.modules['_loss'] = _loss_cython
except Exception:
    pass

# Ensure the root directory is on the python search path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app
