import os
import fire
import fitz

# DEMO = (
#     'You can convert an pdf file into image file by calling `pdf_convert_to_image` with 2 arguments.\n'
#     '1. The path to the pdf file: pdf_file_path: str\n'
#     '2. The path to save the image file: image_file_path: str\n'
#     "You can call it by: {'app': 'pdf', 'action': 'pdf_convert_to_image' , 'pdf_file_path': ... , 'image_file_path': ...}"
# )

DEMO = (
    'convert a pdf file to an image file: '
    '{"app": "pdf", "action": "convert_to_image", "pdf_file_path": [THE_PATH_TO_THE_PDF_FILE], "image_file_path": [THE_PATH_TO_THE_IMAGE_FILE]}'
)


def construct_action(work_dir, args: dict, py_file_path='/apps/pdf_app/pdf_convert_to_image.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --pdf_file_path {args["pdf_file_path"]} --image_file_path {args["image_file_path"]}'


def pdf_convert_to_image(pdf_file_path, image_file_path):
    try:
        # Open the PDF file
        with fitz.open(pdf_file_path) as doc:
            # Assuming you want to convert the first page of the PDF to an image
            page = doc.load_page(0)  # index 0 is the first page

            # Render the page to a pixmap (an image)
            pix = page.get_pixmap()

            # Save the pixmap as an image
            pix.save(image_file_path)

        status = "Success"
    except Exception as e:
        # If there's an error, capture the error message in the status
        status = f"Failed: {str(e)}"

    return status

def main(pdf_file_path, image_file_path):
    if not os.path.exists(pdf_file_path):
        return f'OBSERVATION: The file {pdf_file_path} does not exist. Failed to convert pdf to image.'
    
    status = pdf_convert_to_image(pdf_file_path, image_file_path)
    
    return 'OBSERVATION: ' + status

if __name__ == '__main__':
    fire.Fire(main)