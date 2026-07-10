import os
import fire
import platform

DEMO = (
    'convert an excel document to a pdf: '
    '{"app": "excel", "action": "convert_to_pdf", "excel_file_path": [THE_PATH_TO_THE_EXCEL_FILE], "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE]}'
)

def construct_action(work_dir, args: dict, py_file_path='/apps/excel_app/excel_convert_to_pdf.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --excel_file_path {args["excel_file_path"]} --pdf_file_path {args["pdf_file_path"]}'

    
def linux_doc2pdf(excel_file_path, pdf_file_path):
    try:
        import subprocess
        output_dir = os.path.dirname(pdf_file_path)
        subprocess.call(['libreoffice', '--headless', '--convert-to', 'pdf', excel_file_path, '--outdir', output_dir])
        if pdf_file_path.endswith('.pdf'):
            # check if the filename needs to be changed, move the file if required
            # Get the expected PDF file name based on the Word file name
            base_name = os.path.splitext(os.path.basename(excel_file_path))[0]
            expected_pdf_path = os.path.join(output_dir, base_name + '.pdf')
            
            # Rename the file if the names are different
            if expected_pdf_path != pdf_file_path:
                subprocess.call(['mv', expected_pdf_path, pdf_file_path])
        return True
    except:
        return False

def excel_convert_to_pdf(excel_file_path, pdf_file_path):
    os_name = platform.system()
    return linux_doc2pdf(excel_file_path, pdf_file_path)

def main(excel_file_path, pdf_file_path):
    if not os.path.exists(excel_file_path):
        return f"OBSERVATION: {excel_file_path} does not exist. Failed to convert {excel_file_path} to {pdf_file_path}"
    success = excel_convert_to_pdf(excel_file_path, pdf_file_path)
    if success:
        observation = f"OBSERVATION: Successfully convert {excel_file_path} to {pdf_file_path}"
    else:
        observation = f"OBSERVATION: Failed to convert {excel_file_path} to {pdf_file_path}"
    return observation

if __name__ == '__main__':
    fire.Fire(main)
