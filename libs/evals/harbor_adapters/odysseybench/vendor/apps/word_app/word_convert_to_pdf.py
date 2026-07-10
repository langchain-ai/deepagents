import os
import fire
import platform

# DEMO = (
#     'You can convert a word document to a pdf by calling `word_convert_to_pdf` with 2 arguments.\n'
#     '1. The path of the word document: word_file_path: str\n'
#     '2. The path of the pdf document: pdf_file_path: str\n'
#     "You can call it by: {'app': 'word', 'action': 'convert_to_pdf', 'word_file_path': ..., 'pdf_file_path': ...}"
# )

DEMO = (
    'convert a word document to a pdf: '
    '{"app": "word", "action": "convert_to_pdf", "word_file_path": [THE_PATH_TO_THE_WORD_FILE], "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}'
)



def construct_action(work_dir, args: dict, py_file_path='/apps/word_app/word_convert_to_pdf.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --word_file_path {args["word_file_path"]} --pdf_file_path {args["pdf_file_path"]}'


def mac_win_doc2pdf(word_file_path, pdf_file_path):
    try:
        # Use docx2pdf to convert word to pdf (slow)
        # import docx2pdf
        # docx2pdf.convert(word_file_path, pdf_file_path)

        # Use aspose to convert word to pdf (with watermark)
        import aspose.words as aw
        doc = aw.Document(word_file_path)
        doc.save(pdf_file_path, aw.SaveFormat.PDF)
        return True
    except:
        return False
    
def linux_doc2pdf(word_file_path, pdf_file_path):
    try:
        import subprocess
        output_dir = os.path.dirname(pdf_file_path)
        subprocess.call(['libreoffice', '--headless', '--convert-to', 'pdf', word_file_path, '--outdir', output_dir])
        if pdf_file_path.endswith('.pdf'):
            # check if the filename needs to be changed, move the file if required
            # Get the expected PDF file name based on the Word file name
            base_name = os.path.splitext(os.path.basename(word_file_path))[0]
            expected_pdf_path = os.path.join(output_dir, base_name + '.pdf')
            
            # Rename the file if the names are different
            if expected_pdf_path != pdf_file_path:
                subprocess.call(['mv', expected_pdf_path, pdf_file_path])
        return True
    except:
        return False

def word_convert_to_pdf(word_file_path, pdf_file_path):
    os_name = platform.system()
    if os_name == "Windows" or os_name == "Darwin":
        return mac_win_doc2pdf(word_file_path, pdf_file_path)
    elif os_name == "Linux":
        return linux_doc2pdf(word_file_path, pdf_file_path)
    else:
        raise NotImplementedError

def main(word_file_path, pdf_file_path):
    if not os.path.exists(word_file_path):
        return f"OBSERVATION: The word file {word_file_path} does not exist. Failed to convert the file to pdf."
    
    success = word_convert_to_pdf(word_file_path, pdf_file_path)
    if success:
        observation = f"OBSERVATION: Successfully convert {word_file_path} to {pdf_file_path}"
    else:
        observation = f"OBSERVATION: Failed to convert {word_file_path} to {pdf_file_path}"
    return observation

if __name__ == '__main__':
    fire.Fire(main)
