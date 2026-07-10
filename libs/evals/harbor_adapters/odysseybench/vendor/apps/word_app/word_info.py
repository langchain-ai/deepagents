from apps.word_app.word_create_new_file import DEMO as create_new_file_demo
from apps.word_app.word_read_file import DEMO as read_file_demo
from apps.word_app.word_write_to_file import DEMO as write_to_file_demo
from apps.word_app.word_convert_to_pdf import DEMO as convert_to_pdf_demo

class WordInfo:
    def __init__(self):
        self.name = "word"
        self.info = {
            "create_new_file": create_new_file_demo,
            "read_file": read_file_demo,
            "write_to_file": write_to_file_demo,
            "convert_to_pdf": convert_to_pdf_demo,
        }

    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)