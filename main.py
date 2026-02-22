import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene
)
import pyqtgraph as pg
from PySide6.QtGui import QBrush, QPen
from PySide6.QtCore import Qt, QTimer



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drilling Visualization")
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        control_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.start_button.clicked.connect(self.start_animation)
        self.pause_button.clicked.connect(self.pause_animation)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)

        main_layout.addLayout(control_layout)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        content_layout.addWidget(self.view, 2)

        self.draw_drill()
        self.depth = 0
        self.rotation_angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)

        graphs_layout = QVBoxLayout()

        self.graph1 = pg.PlotWidget(title="Drilling Speed")
        self.graph2 = pg.PlotWidget(title="Rotation Speed")

        graphs_layout.addWidget(self.graph1)
        graphs_layout.addWidget(self.graph2)

        content_layout.addLayout(graphs_layout, 1)

    def draw_drill(self):
        # Мачта
        self.scene.addRect(150, 50, 40, 300, QPen(Qt.black), QBrush(Qt.darkGray))

        # Буровой став (линия)
        self.drill_pipe = self.scene.addLine(170, 350, 170, 600, QPen(Qt.blue, 6))

        # Долото (круг)
        self.drill_bit = self.scene.addEllipse(150, 600, 40, 40, QPen(Qt.red), QBrush(Qt.red))
    
    def start_animation(self):
        self.timer.start(50)  # 20 кадров в секунду

    def pause_animation(self):
        self.timer.stop()
    def update_animation(self):
        # Увеличиваем глубину
        self.depth += 1

        # Двигаем бур вниз
        self.drill_pipe.setLine(170, 350, 170, 600 + self.depth)
        self.drill_bit.setRect(150, 600 + self.depth, 40, 40)

        # Вращаем долото
        self.rotation_angle += 10
        self.drill_bit.setRotation(self.rotation_angle)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    