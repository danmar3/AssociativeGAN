import os
import shutil


def clean_folder(data_dir='tmp/gmmgan'):
    '''delete folders with no images or checkpoints'''
    session_folders = sorted(
        os.path.join(data_dir, folder)
        for folder in os.listdir(data_dir)
        if 'session' in folder)
    empty_folders = [folder for folder in session_folders
                     if 'checkpoints' not in os.listdir(folder)]
    print('removing empty folders: {}'.format(empty_folders))
    for folder in empty_folders:
        shutil.rmtree(folder)
