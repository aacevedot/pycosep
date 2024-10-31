import os
import tempfile


class RuntimeSettings:
    def __init__(self):
        # self.root_path = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda: 0)))
        self.root_path = os.path.join(tempfile.gettempdir(), "pycosep")
        self._create_directory_if_not_exists(self.root_path)

        self.concorde_executable = 'concorde'

        if os.name == 'nt':
            self.concorde_executable += '.exe'
            self.concorde_path = os.path.join('C:\\cygwin32\\bin\\', self.concorde_executable)
        else:
            self.concorde_path = os.path.join(self.root_path, 'bin', self.concorde_executable)

        if not os.path.isfile(self.concorde_path):
            raise RuntimeError(f'concorde executable not found in \'{self.concorde_path}\'')

        # self.temp_path = os.path.join(self.root_path, '.temp')
        self.temp_path = self.root_path
        self._create_directory_if_not_exists(self.temp_path)

    def _create_directory_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
