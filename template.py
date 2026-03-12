import os
from pathlib import Path
import logging
import json

# -------------------- logging setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# -------------------- project files --------------------
list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "test.py"
]

# -------------------- optional starter content --------------------
file_contents = {
    "src/__init__.py": "",
    "src/helper.py": '''def hello_helper():
    return "helper file created successfully"
''',
    "src/prompt.py": '''system_prompt = "You are a helpful assistant."
''',
    ".env": '''# Add your environment variables here
# OPENAI_API_KEY=your_key_here
''',
    "setup.py": '''from setuptools import setup, find_packages

setup(
    name="medical-chatbot",
    version="0.0.1",
    author="Your Name",
    packages=find_packages(),
    install_requires=[],
)
''',
    "app.py": '''from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Medical Chatbot App is running!"

if __name__ == "__main__":
    app.run(debug=True)
''',
    "test.py": '''def test_sample():
    assert True
''',
}

# valid empty notebook structure
empty_notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}


def create_project_structure(files: list[str], base_path: Path = Path.cwd()) -> None:
    """
    Create folders and files for the project structure.
    If a file already exists, it is skipped.
    """
    for file in files:
        file_path = base_path / file

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # If file does not exist, create it
            if not file_path.exists():
                if file_path.suffix == ".ipynb":
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(empty_notebook, f, indent=4)
                else:
                    content = file_contents.get(file, "")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                logging.info(f"Created file: {file_path}")
            else:
                logging.info(f"File already exists, skipped: {file_path}")

        except Exception as e:
            logging.error(f"Error creating {file_path}: {e}")


if __name__ == "__main__":
    logging.info(f"Current working directory: {os.getcwd()}")
    create_project_structure(list_of_files)
    logging.info("Project structure created successfully.")