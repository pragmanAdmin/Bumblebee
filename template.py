import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",
    "api/main.py",
    "api/models.py",
    "api/routes.py",
    "api/__init__.py",

    "docker/Dockerfile",
    "docker/docker-compose.yml",

    "notebooks/product_attr_model.ipynb",

    "scripts/train.py",
    "scripts/inference.py",
    "scripts/utils.py",
    "scripts/dataset.py",
    "scripts/transformer.py",
    "scripts/__init__.py",

    "tests/test_api.py",
    "tests/test_model.py",

    "config/config.yaml",
    "config/__init__.py",

    ".env",
    "requirements.txt"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
            
    else:
        logging.info(f"{filename} already exists")
