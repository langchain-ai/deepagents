import os
import fire
from pdf2docx import Converter

# DEMO = (
#     'You can convert an pdf file into word file by calling `pdf_convert_to_word` with 2 arguments.\n'
#     '1. The path to the pdf file: pdf_file_path: str\n'
#     '2. The path to save the word file: word_file_path: str\n'
#     "You can call it by: {'app': 'pdf', 'action': 'pdf_convert_to_word', 'pdf_file_path': ... , 'word_file_path': ...}"
# )

DEMO = (
    'convert a pdf file to a word file: '
    '{"app": "pdf", "action": "convert_to_word", "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE], "word_file_path": [THE_PATH_TO_THE_WORD_FILE]}'
)


def construct_action(work_dir, args: dict, py_file_path='/apps/pdf_app/pdf_convert_to_word.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --pdf_file_path {args["pdf_file_path"]} --word_file_path {args["word_file_path"]}'


def pdf_convert_to_word(pdf_file_path, word_file_path):
    try:
        # Initialize the converter
        cv = Converter(pdf_file_path)

        # Convert all pages of the PDF and save it as a Word document
        cv.convert(word_file_path, start=0, end=None)
        cv.close()

        status = "Success"
    except Exception as e:
        # If there's an error, capture the error message in the status
        status = f"Failed: {str(e)}"

    return status

def main(pdf_file_path, word_file_path):
    if not os.path.exists(pdf_file_path):
        return f"OBSERVATION: The pdf file {pdf_file_path} does not exist. Failed to convert the file to word."

    status = pdf_convert_to_word(pdf_file_path, word_file_path)
    
    return 'OBSERVATION: ' + status

if __name__ == '__main__':
    fire.Fire(main)