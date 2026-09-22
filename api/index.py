import os
import sys

# Ensure root directory is added to sys.path so app and its assets can be located
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Vercel entrypoint
handler = app
