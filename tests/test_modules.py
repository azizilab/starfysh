import unittest
import importlib

class TestModules(unittest.TestCase):
    def test_import_modules(self):
        modules = [
            "AA",
            "dataloader",
            "plot_utils",
            "post_analysis",
            "starfysh",
            "utils",
            "utils_integrate",
        ]

        for module_name in modules:
            try:
                importlib.import_module("starfysh." + module_name)
                print(f"Module {module_name} imported successfully.")
            except ImportError as e:
                self.fail(f"Failed to import module {module_name}: {e}")

if __name__ == '__main__':
    unittest.main()
