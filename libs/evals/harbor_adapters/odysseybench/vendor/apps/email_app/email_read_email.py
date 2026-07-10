import os
import fire
from email import policy
from email.parser import BytesParser

DEMO = (
    'Read a user\'s email by the given Email ID: '
    '{"app": "email", "action": "read_email", "username": [USERNAME], "email_id": [EMAIL_ID]}'
)

def construct_action(word_dir, args: dict, py_file_path='/apps/email_app/email_read_email.py'):
    return "python3 {} --email_id '''{}''' --username '''{}'''".format(
        py_file_path,
        args["email_id"],
        args["username"],
    )

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


def read_email(username, email_id, testbed_dir='/testbed'):
    """
    Read a user's email by the given Email ID.
    """
    os.makedirs(f'{testbed_dir}/emails/{username}', exist_ok=True)
    try:
        if not email_id.endswith('.eml'):
            email_id += '.eml'
        email_file = f'{testbed_dir}/emails/{username}/{email_id}'
        with open(email_file, 'rb') as f:
            email_content = f.read()
            email = BytesParser(policy=policy.default).parsebytes(email_content)
        message = ''
        message += f'From: {email["From"]}\n'
        message += f'To: {email["To"]}\n'
        message += f'Subject: {email["Subject"]}\n'
        message += f'Content: {get_email_content(email) + "..."}\n'
        return message
    except Exception as e:
        print('!!!', e)
        return 'Error: Failed to read email.'


def main(username, email_id, testbed_dir='/testbed'):
    message = read_email(username, email_id, testbed_dir=testbed_dir)
    if message != 'Error: Failed to read email.':
        observation = f"OBSERVATION: Successfully read email {email_id} for {username}:\n{message}"
    else:
        observation = f"OBSERVATION: Failed to read email {email_id} for {username}."
    return observation


if __name__ == '__main__':
    fire.Fire(main)
