import os
import unittest

from pycosep.runtime_settings import RuntimeSettings


class TestRuntimeSettings(unittest.TestCase):
    def test_runtime_settings_raises_value_error_when_invalid_concorde_path(self):
        concorde_path = 1234  # invalid
        self.assertRaises(ValueError, RuntimeSettings, concorde_path)

    def test_runtime_settings_raises_runtime_error_when_concorde_path_not_found(self):
        concorde_path = "/not/existing/path/to/concorde"
        self.assertRaises(RuntimeError, RuntimeSettings, concorde_path)

    def test_runtime_settings_passes_when_default_settings(self):
        settings = RuntimeSettings()
        if os.name == 'nt':
            self.assertEqual(settings.concorde_path, 'C:\\cygwin32\\bin\\concorde.exe')
        else:
            self.assertEqual(settings.concorde_path, '/usr/local/bin/concorde')

        self.assertRegex(settings.temp_path, r'pycosep')
