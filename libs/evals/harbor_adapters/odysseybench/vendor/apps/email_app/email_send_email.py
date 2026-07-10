import os
import fire
from email.message import EmailMessage

DEMO = (
    'Send an email to a recipient: '
    '{"app": "email", "action": "send_email", "sender": [SENDER], "recipient": [RECIPIENT], "subject": [SUBJECT], "content": [CONTENT]}'
)

def construct_action(word_dir, args: dict, py_file_path='/apps/email_app/email_send_email.py'):
    if isinstance(args["recipient"], list):
        args["recipient"] = 'Multiple recipients'
    return """python3 {} --sender {} --recipient '''{}''' --subject '''{}''' --content '''{}''' """.format(
        py_file_path,
        args["sender"],
        args["recipient"],
        args["subject"].replace("'", "").replace('"', ''),
        args["content"].replace("'", "").replace('"', '')
    )

def send_email(sender, recipient, subject, content):
    """
    Send an email from a user to a recipient.
    """
    if '@' in sender:
        sender = sender.split('@')[0]
    if '@' in recipient:
        recipient = recipient.split('@')[0]

    os.makedirs(f'/testbed/emails/{sender}', exist_ok=True)
    os.makedirs(f'/testbed/emails/{recipient}', exist_ok=True)
    try:
        email = EmailMessage()
        email['From'] = sender + '@example.com'
        email['To'] = recipient + '@example.com'
        email['Subject'] = subject
        email.set_content(content)
        email_file = f'/testbed/emails/{recipient}/{subject}.eml'
        with open(email_file, 'w') as f:
            f.write(email.as_string())
        email_file = f'/testbed/emails/{sender}/{subject}.eml'
        with open(email_file, 'w') as f:
            f.write(email.as_string())    
        return f'Successfully sent email to {recipient}.'
    except Exception as e:
        print('!!!', e)
        return 'Error: Failed to send email.'

def main(sender, recipient, subject, content):
    if recipient == 'Multiple recipients':
        observation = f"OBSERVATION: Failed to send email to {recipient}. Only support one recipient."
        return observation
    message = send_email(sender, recipient, subject, content)
    if message == 'Error: Failed to send email.':
        observation = f"OBSERVATION: Failed to send email to {recipient}."
    else:
        observation = f"OBSERVATION: Successfully sent email to {recipient}."
    return observation


if __name__ == '__main__':
    fire.Fire(main)
