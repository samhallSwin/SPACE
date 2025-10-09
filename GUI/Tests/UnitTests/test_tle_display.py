import sys
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

repo_root = Path(file).parents[3]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "GUI"))

from GUI.Components.TLEDisplay import TLEDisplay
class HostWindow(QWidget):
    """Top-level owner so Qt destroys children cleanly during teardown."""
    def init(self):
        super().init()
        self.setWindowFlag(Qt.Tool)
        self.backend = None
        self.layout = QVBoxLayout(self)
        self.tle_display = TLEDisplay(parent=self)
        self.layout.addWidget(self.tle_display)


def _force_state(tle_display: TLEDisplay, *, expanded: bool, qtbot):
    """
    Drive the widget using its public toggle until tledisplay.expanded == expanded.
    (We don't set the flag directly; we only use the public control.)
    """
    # Try up to 2 toggles to land on desired state
    for in range(2):
        if tle_display.expanded == expanded:
            break
        tle_display.toggle_size()
        qtbot.wait(10)
    assert tle_display.expanded == expanded, "Could not reach requested expand/collapse state"


def test_drop_label_disabled_when_collapsed(qtbot):
    """DropLabel must NOT accept drops when panel is collapsed (no code changes needed)."""
    host = HostWindow()
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)

    d = host.tle_display

    _force_state(d, expanded=False, qtbot=qtbot)

    # Expect DropLabel to mirror the expanded state
    assert not d.drop_label.acceptDrops(), "DropLabel should be disabled when panel is collapsed"


def test_drop_label_enabled_when_expanded(qtbot):
    """DropLabel SHOULD accept drops when panel is expanded (no code changes needed)."""
    host = HostWindow()
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)

    d = host.tle_display

    _force_state(d, expanded=True, qtbot=qtbot)

    # Expect DropLabel to mirror the expanded state
    assert d.drop_label.acceptDrops(), "DropLabel should be enabled when panel is expanded"