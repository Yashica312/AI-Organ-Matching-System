import os

PORT = int(os.environ.get("PORT", 8501))

from model.app import *
