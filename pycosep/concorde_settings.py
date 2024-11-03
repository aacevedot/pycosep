import os
import tempfile


class ConcordeSettings:
    def __init__(self, concorde_path=None):
        if concorde_path is not None:
            self.concorde_path = concorde_path
        else:
            self._set_default_concorde_path()

        self._validate_concorde_path()

        self.temp_path = os.path.join(tempfile.gettempdir(), "pycosep")
        self._create_directory_if_not_exists(self.temp_path)

    def _validate_concorde_path(self):
        if not isinstance(self.concorde_path, str) or not os.path.isabs(self.concorde_path):
            raise ValueError("'concorde_path' must be a valid absolute path.")

        if not os.path.isfile(self.concorde_path):
            raise RuntimeError(f'concorde executable not found in \'{self.concorde_path}\'')

    def _set_default_concorde_path(self):
        self._concorde_executable = 'concorde'

        if os.name == 'nt':
            self._concorde_executable += '.exe'
            self.concorde_path = os.path.join('C:\\cygwin32\\bin\\', self._concorde_executable)
        else:
            self.concorde_path = os.path.join('/usr/local/bin', self._concorde_executable)

    def _create_directory_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
