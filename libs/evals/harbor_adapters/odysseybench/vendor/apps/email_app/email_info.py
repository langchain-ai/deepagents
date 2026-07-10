from apps.email_app.email_list_emails import DEMO as list_emails_demo
from apps.email_app.email_read_email import DEMO as read_email_demo
from apps.email_app.email_send_email import DEMO as send_email_demo

class EmailInfo:
    def __init__(self):
        self.name = "email"
        self.info = {
            "list_emails": list_emails_demo,
            "read_email": read_email_demo,
            "send_email": send_email_demo,
        }
    
    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)