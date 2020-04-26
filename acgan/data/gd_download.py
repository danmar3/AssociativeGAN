import pickle
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request



class Downloader(object):
    def __init__(self):
        API_KEY = ''
        self.service = build('drive', 'v3', developerKey=API_KEY)

    def listdir(self, folder_id):
        response = self.service.files()\
            .list(q="'{}' in parents".format(folder_id))\
            .execute()
        files = response['files']
        return files

    def download_file(self, file_id, filename):
        request = self.service.files().get_media(fileId=file_id)
        with open(filename, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print("Download %d%%." % int(status.progress() * 100))

    def download_folder(self, folder_id):
        files = [file for file in self.listdir(folder_id)
                 if file['kind'] == 'drive#file']
        for file_info in files:
            self.download_file(file_info['id'], file_info['name'])
