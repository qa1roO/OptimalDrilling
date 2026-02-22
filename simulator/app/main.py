import sys

from PySide6.QtWidgets import QApplication

from .window import SimulatorMainWindow


def run() -> int:
    app = QApplication(sys.argv)
    window = SimulatorMainWindow()
    window.show()
    return app.exec()
