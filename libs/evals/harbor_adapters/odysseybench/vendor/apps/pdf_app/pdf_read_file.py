import os
import fire
from PyPDF2 import PdfReader 

# DEMO = (
#     'You can read a pdf file by calling `pdf_read_file` with 1 argument.\n'
#     '1. The path to the pdf file: pdf_file_path: str\n'
#     "You can call it by: {'app': 'pdf', 'action': 'pdf_read_file', 'pdf_file_path': ...}"
# )

DEMO = (
    'read a pdf file: '
    '{"app": "pdf", "action": "read_file", "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}'
)

def construct_action(work_dir, args: dict, py_file_path='/apps/pdf_app/pdf_read_file.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --pdf_file_path {args["pdf_file_path"]}'


def read_pdf(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    return reader.pages

def read_pdf_to_string(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    pages = reader.pages
    text = ""
    # getting a specific page from the pdf file
    for i in range(len(pages)):
        page = pages[i] 
        text += page.extract_text()
    return text

def wrap(pages):
    print(f"number of pages: {len(pages)}") 
  
    text = ""
    # getting a specific page from the pdf file
    for i in range(len(pages)):
        page = pages[i] 
        text += page.extract_text()
    
    return text
    
    

def main(pdf_file_path, debug=False):
    if not os.path.exists(pdf_file_path):
        return f"OBSERVATION: The pdf file {pdf_file_path} does not exist. Failed to read the file."
    pages = read_pdf(pdf_file_path)
    if debug:
        print(pages)
    return 'OBSERVATION: ' + wrap(pages)

if __name__ == '__main__':
    fire.Fire(main)