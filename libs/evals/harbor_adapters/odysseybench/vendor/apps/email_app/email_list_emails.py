import fire
import os
from glob import glob
from email import policy
from email.parser import BytesParser

DEMO = (
    'List emails for a given username: '
    '{"app": "email", "action": "list_emails", "username": [USER_NAME]}'
)

def construct_action(word_dir, args: dict, py_file_path='/apps/email_app/email_list_emails.py'):
    return f'python3 {py_file_path} --username {args["username"]}'


def get_email_content(msg):
    if msg.is_multipart():
        parts = []
        for part in msg.iter_parts():
            if part.get_content_type() == 'text/plain':
                parts.append(part.get_payload(decode=True).decode(part.get_content_charset(), errors="replace"))
            elif part.get_content_type() == 'text/html':
                parts.append(part.get_payload(decode=True).decode(part.get_content_charset(), errors="replace"))
        return "\n".join(parts)
    else:
        return msg.get_payload(decode=True).decode(msg.get_content_charset(), errors="replace")


def list_emails(username, testbed_dir='/testbed', first_n_letters=20):
    """
    List emails for a given username.
    """
    os.makedirs(f'{testbed_dir}/emails/{username}', exist_ok=True)
    try:
        email_folder = f'{testbed_dir}/emails/{username}'
        email_files = glob(f'{email_folder}/*.eml')
        emails = []
        for email_file in email_files:
            with open(email_file, 'rb') as f:
                email_content = f.read()
                email = BytesParser(policy=policy.default).parsebytes(email_content)
                emails.append(email)
        message = ''
        for email, email_file in zip(emails, email_files):
            email_name = email_file.split('/')[-1]
            message += f'Email ID: {email_name}\n'
            message += f'From: {email["From"]}\n'
            message += f'To: {email["To"]}\n'
            message += f'Subject: {email["Subject"]}\n'
            if first_n_letters != -1:
                message += f'Content: {get_email_content(email)[:first_n_letters] + "..."}\n'
            else:
                message += f'Content: {get_email_content(email)}\n'
            message += '-'*50 + '\n'
        if not message:
            message = 'No emails found.'
        return message
    except Exception as e:
        return 'Error: Failed to list emails.'



def main(username, testbed_dir='/testbed'):
    message = list_emails(username, testbed_dir=testbed_dir)
    if message == 'No emails found.':
        observation = f"OBSERVATION: No emails found for {username}."
    elif message != 'Error: Failed to list emails.':
        observation = f"OBSERVATION: Successfully list emails for {username}:\n{message}"
    else:
        observation = f"OBSERVATION: Failed to list emails for {username}."
    return observation


if __name__ == '__main__':
    fire.Fire(main)

