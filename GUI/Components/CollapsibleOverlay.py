from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, Qt, QSize
from PyQt5.QtWidgets import QPushButton, QWidget, QFrame, QHBoxLayout, QVBoxLayout, QGraphicsOpacityEffect
from enum import Enum

class OverlaySide(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4

class CollapsibleOverlay(QFrame):
    def __init__(self, parent=None, side=OverlaySide.LEFT):
        super().__init__(parent)

        self.side = side
        self.expanded = False

        self.full_size = 450   # Expanded width/height
        self.collapsed_size = 20
        self.above_height_amount = 0
        self.control_panel_height_reduction = 110

        # Overlay content container
        self.overlay = QWidget(self)
        self.overlay.setStyleSheet("background-color: #dddddd;")

        self.content_layout = QVBoxLayout(self.overlay)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        # Fade effect
        self.opacity_effect = QGraphicsOpacityEffect(self.overlay)
        self.overlay.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

        # Decide orientation based on side
        if self.side in (OverlaySide.LEFT, OverlaySide.RIGHT):
            self.main_layout = QHBoxLayout(self)
            self.toggle_button = QPushButton("▶" if self.side == OverlaySide.LEFT else "◀")
            self.toggle_button.setFixedWidth(self.collapsed_size)

            if self.side == OverlaySide.LEFT:
                self.main_layout.addWidget(self.toggle_button)
                self.main_layout.addWidget(self.overlay, stretch=1)
            else:
                self.main_layout.addWidget(self.overlay, stretch=1)
                self.main_layout.addWidget(self.toggle_button)

        else:  # TOP or BOTTOM
            self.main_layout = QVBoxLayout(self)
            self.toggle_button = QPushButton("▼" if self.side == OverlaySide.TOP else "▲")
            self.toggle_button.setFixedHeight(self.collapsed_size)

            if self.side == OverlaySide.TOP:
                self.main_layout.addWidget(self.toggle_button)
                self.main_layout.addWidget(self.overlay, stretch=1)
            else:
                self.main_layout.addWidget(self.overlay, stretch=1)
                self.main_layout.addWidget(self.toggle_button)

        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toggle_button.setAccessibleName("button_ExpandCollapse")
        
        self.toggle_button.clicked.connect(self.toggle_size)
        self.update_size()

    def toggle_size(self):
        self.expanded = not self.expanded
        self.animate_size()
        self.animate_fade()
        self.update_button_icon()

    def update_button_icon(self):
        if self.side == OverlaySide.LEFT:
            self.toggle_button.setText("◀" if self.expanded else "▶")
        elif self.side == OverlaySide.RIGHT:
            self.toggle_button.setText("▶" if self.expanded else "◀")
        elif self.side == OverlaySide.TOP:
            self.toggle_button.setText("▲" if self.expanded else "▼")
        else:  # BOTTOM
            self.toggle_button.setText("▼" if self.expanded else "▲")

    def update_size(self):
        if self.side in (OverlaySide.LEFT, OverlaySide.RIGHT):
            target_width = self.full_size if self.expanded else self.collapsed_size
            target_height = self.parent().height() - self.above_height_amount - self.control_panel_height_reduction
            self.setFixedSize(target_width, target_height)
            self.toggle_button.setFixedHeight(target_height)
        else:  # TOP or BOTTOM
            target_height = self.full_size if self.expanded else self.collapsed_size
            target_width = self.parent().width()
            self.setFixedSize(target_width, target_height)
            self.toggle_button.setFixedWidth(target_width)

        self.reposition()

    def animate_size(self):
        if self.side in (OverlaySide.LEFT, OverlaySide.RIGHT):
            start_width = self.width()
            end_width = self.full_size if self.expanded else self.collapsed_size

            animation = QPropertyAnimation(self, b"minimumWidth")
            animation.setDuration(300)
            animation.setStartValue(start_width)
            animation.setEndValue(end_width)
            animation.setEasingCurve(QEasingCurve.InOutCubic)
            animation.start()
            self._animation = animation
        else:  # TOP or BOTTOM
            start_height = self.height()
            end_height = self.full_size if self.expanded else self.collapsed_size

            animation = QPropertyAnimation(self, b"minimumHeight")
            animation.setDuration(300)
            animation.setStartValue(start_height)
            animation.setEndValue(end_height)
            animation.setEasingCurve(QEasingCurve.InOutCubic)
            animation.start()
            self._animation = animation

    def animate_fade(self):
        end_opacity = 1.0 if self.expanded else 0.0
        fade = QPropertyAnimation(self.opacity_effect, b"opacity")
        fade.setDuration(300)
        fade.setStartValue(self.opacity_effect.opacity())
        fade.setEndValue(end_opacity)
        fade.setEasingCurve(QEasingCurve.InOutCubic)
        fade.start()
        self._fade = fade

    def reposition(self):
        """Dock overlay to the correct side of the parent"""
        parent = self.parent()
        if not parent:
            return

        if self.side == OverlaySide.LEFT:
            self.move(0, self.above_height_amount)
        elif self.side == OverlaySide.RIGHT:
            self.move(parent.width() - self.width(), self.above_height_amount)
        elif self.side == OverlaySide.TOP:
            self.move(0, 0)
        elif self.side == OverlaySide.BOTTOM:
            self.move(0, parent.height() - self.height())

    def resizeEvent(self, event):
        """Keep overlay docked when parent resizes"""
        self.update_size()
        super().resizeEvent(event)
