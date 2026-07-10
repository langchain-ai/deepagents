import os
import fire
import openpyxl

DEMO = (
    'create a new excel file: '
    '{"app": "excel", "action": "create_new_file", "file_path": [THE_PATH_TO_THE_NEW_EXCEL_FILE]}'
)

def construct_action(work_dir, args: dict, py_file_path='/apps/excel_app/excel_create_new_file.py'):
    return f'python3 {py_file_path} --file_path {args["file_path"]}'


def excel_create_new_file(file_path):
    try:
        wb = openpyxl.Workbook()
        wb.save(file_path)
        return True
    except:
        return False
    
def main(file_path):
    if os.path.exists(file_path):
        observation = f"File {file_path} already exists"
        return 'OBSERVATION: ' + observation
    success = excel_create_new_file(file_path)
    if success:
        observation = f"Successfully create new file {file_path}"
    else:
        observation = f"Failed to create new file {file_path}"
    return 'OBSERVATION: ' + observation

if __name__ == '__main__':
    fire.Fire(main)
