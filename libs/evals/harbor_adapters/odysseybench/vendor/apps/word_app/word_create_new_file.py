import fire
import pandas as pd
from docx import Document

# DEMO = (
#     'You can create a new word file by calling `word_create_new_file` with 1 arguments.\n'
#     '1. The path of the new word file: file_path: str\n'
#     "You can call it by: {'app': 'word', 'action': 'word_create_new_file', 'file_path': ...}"
# )

DEMO = (
    'create a new word file: '
    '{"app": "word", "action": "create_new_file", "file_path": [THE_PATH_TO_THE_NEW_WORD_FILE]}'
)



def construct_action(work_dir, args: dict, py_file_path='/apps/word_app/word_create_new_file.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --file_path {args["file_path"]}'

def word_create_new_file(file_path):
    try:
        document = Document()
        document.save(file_path)
        return True
    except:
        return False

def main(file_path):
    success = word_create_new_file(file_path)
    if success:
        observation = f"Successfully create new file {file_path}"
    else:
        observation = f"Failed to create new file {file_path}"
    return 'OBSERVATION: ' + observation

if __name__ == '__main__':
    fire.Fire(main)
