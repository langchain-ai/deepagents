from apps.pdf_app.image_convert_to_pdf import DEMO as convert_to_pdf_demo
from apps.pdf_app.pdf_convert_to_image import DEMO as convert_to_image_demo
from apps.pdf_app.pdf_read_file import DEMO as read_file_demo
from apps.pdf_app.pdf_convert_to_word import DEMO as convert_to_word_demo

class PDFInfo:
    def __init__(self):
        self.name = "pdf"
        self.info = {
            "convert_to_pdf": convert_to_pdf_demo,
            "convert_to_image": convert_to_image_demo,
            "read_file": read_file_demo,
            "convert_to_word": convert_to_word_demo,
        }
    
    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)