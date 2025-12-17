import os
import re
import json

def remove_markdown_boundary(text):
    return re.sub(r'^```json\n(.*)\n```$', r'\1', text.strip(), flags=re.DOTALL)

def get_files(directory, extension='.csv'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def filename(path):
    return os.path.splitext(os.path.basename(path))[0]