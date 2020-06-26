import os
import getpass
import subprocess


class RemoteStorage(object):
    REMOTE_IP = "128.172.184.42"
    REMOTE_FOLDER = "/data/shared/research/acgan"

    def __init__(self, username=None):
        raise NotImplementedError(
            'This is not working. Use RemoteShare instead.')
        if username is None:
            username = getpass.getuser()
        self.ssh = paramiko.SSHClient()
        # automatically add keys without requiring human intervention
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.ssh.connect(self.REMOTE_IP, username=username, password=sftpPass)

    def listdir(self):
        with self.ssh.open_sftp() as ftp:
            files = ftp.listdir()
        return files

    def upload(self, source):
        pass

    def download(self, source):
        assert source in self.listdir(), 'source not found.'


class RemoteShare(object):
    """ Download and keep track of models saved remotely.
    Models are saved in a remote shared folder which is divided by username,
    followed by the model name.
    """

    REMOTE_IP = '128.172.184.42'
    _remote_base_path = '/data/shared/research/acgan/'

    @staticmethod
    def ssh_list(username, ipaddr, path):
        """ list models stored in the remote shared folder.
        Args:
            username: user name for autentication.
            ipaddr: ip address of remote.
            path: remote path.
        """
        process = subprocess.Popen(
            ["ssh", f"{username}@{ipaddr}", "ls", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        files = [si for si in stdout.decode().split('\n') if si]
        return files

    @classmethod
    def list_remote_models(cls, auth_user=None):
        if auth_user is None:
            auth_user = getpass.getuser()
        unames = cls.ssh_list(auth_user, cls.REMOTE_IP, cls._remote_base_path)
        for user in unames:
            models = cls.ssh_list(auth_user, cls.REMOTE_IP,
                                  os.path.join(cls._remote_base_path, user))
            print("{}: {}".format(user, models))

    def __init__(self, model, username=None, auth_user=None):
        """Download and keep track of models saved in remote shared folder
        Args:
            model: name of the model. To list available models use
            list_remote_models.
            username: user name folder.
            auth_user: user to authenticate in remote server.
        """
        if username is None:
            username = getpass.getuser()
        if auth_user is None:
            auth_user = getpass.getuser()
        self._model = model
        self._username = username
        self._auth_user = auth_user
        self._local_base_path = 'tmp/saved_sessions'

    def get_local_path(self, session=None):
        path = os.path.join(
            self._local_base_path, self._username, self._model)
        if session is not None:
            path = os.path.join(path, session)
        return path

    def get_remote_path(self, session=None):
        path = os.path.join(
            self._remote_base_path, self._username, self._model)
        if session is not None:
            path = os.path.join(path, session)
        return path

    def list_local(self, identifier=None):
        local_path = os.path.join(
            self._local_base_path, self._username, self._model)

        if not os.path.exists(local_path):
            return []

        sessions = os.listdir(local_path)
        if identifier is None:
            return sessions
        else:
            return [sess for sess in sessions if identifier in sess]

    def list_remote(self, identifier=None):
        remote_path = self.get_remote_path()

        sessions = self.ssh_list(
            self._auth_user, self.REMOTE_IP, remote_path)

        if identifier is None:
            return sessions
        else:
            return [sess for sess in sessions if identifier in sess]

    def local_exists(self, session):
        return session in self.list_local()

    def remote_exists(self, session):
        return session in self.list_remote()

    def download(self, session, force=False):
        if not force and self.local_exists(session):
            return self.get_local_path(session)

        if not self.remote_exists(session):
            raise ValueError(f"session {session} does not exist.")

        if not os.path.exists(self.get_local_path()):
            os.makedirs(self.get_local_path())
        remote_path = self.get_remote_path(session)
        local_path = self.get_local_path(session)
        process = subprocess.Popen(
            ["scp", "-r",
             f"{self._auth_user}@{self.REMOTE_IP}:{remote_path}", local_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        process.wait()
        return local_path
