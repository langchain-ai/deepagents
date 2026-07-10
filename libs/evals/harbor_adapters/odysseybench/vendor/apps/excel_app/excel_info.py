from apps.excel_app.excel_create_new_file import DEMO as create_file_demo
from apps.excel_app.excel_read_file import DEMO as read_file_demo
from apps.excel_app.excel_set_cell import DEMO as set_cell_demo
from apps.excel_app.excel_delete_cell import DEMO as delete_cell_demo
from apps.excel_app.excel_convert_to_pdf import DEMO as convert_to_pdf_demo

class ExcelInfo:
    def __init__(self):
        self.name = "excel"
        self.info = {
            "create_file": create_file_demo,
            "read_file": read_file_demo,
            "set_cell": set_cell_demo,
            "delete_cell": delete_cell_demo,
            "convert_to_pdf": convert_to_pdf_demo,
        }

    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)