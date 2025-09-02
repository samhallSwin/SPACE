#Set simulation tle_data to whatever is returned from get_data_from_enabled_tle_slots().
#when instancing TLE_Display, connect your update visuals code using tle_display.update_signal.connect('function name here')

#use the following code/function in __init__ to instantiate a copy
#def set_up_tle_display(self):
#        self.tle_display = TLEDisplay(self)
#        self.tle_display.raise_()
#       
#        #Assumes a reference to the sat_sim has been established.
#        self.tle_display.add_elements(self.simulation.tle_data)

#TODO: clean up and attribution pass
#TODO: figure out how to access the SatSimHandler, to access the read_tle_file() function. once figured out, can remove the duplicated function from this code
#TODO:

from Components.CollapsibleOverlay import CollapsibleOverlay
from Components.DropLabel import DropLabel
from Components.TLESlot import TLESlot

from PyQt5.QtWidgets import  QHBoxLayout, QLabel, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtCore import pyqtSignal

class TLEDisplay(CollapsibleOverlay):
    update_signal = pyqtSignal()

    def __init__(self, parent=None):
        self.backend = parent.backend

        self.tle_dict = {}

        super().__init__(parent)

        self.set_up_drop_label()
        self.set_up_scroll_box()

    def set_up_drop_label(self):
        # Use a horizontal layout
        self.drop_layout = QHBoxLayout()
        self.drop_layout.setContentsMargins(10, 10, 10, 10)
        self.drop_layout.setSpacing(15)

        # Add DropLabel on the left
        self.drop_label = DropLabel(self)
        self.drop_layout.addWidget(self.drop_label)

        # Add your existing content on the right
        self.right_content = QLabel("Collapsible content goes here.")
        self.right_content.setWordWrap(True)
        self.drop_layout.addWidget(self.right_content)

        self.content_layout.addLayout(self.drop_layout)

        #set up event listener
        self.drop_label.fileDropped.connect(self.on_file_dropped)
 
    def set_up_scroll_box(self):
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        self.inner_layout = QVBoxLayout(scroll_widget)
        self.scroll_area.setWidget(scroll_widget)

        # Add scroll area to fill remaining space
        self.content_layout.addWidget(self.scroll_area, stretch=1)

    def populate_scroll_items(self):
        #clear all existing items
        #TODO: update existing items, then Add/Clear additional items as needed.
        for i in reversed(range(self.inner_layout.count())):
            widget_to_remove = self.inner_layout.itemAt(i).widget()
            
            #self.inner_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
            widget_to_remove.deleteLater()

        for i in self.tle_dict:
            name = str(i)
            line_1 = self.tle_dict[name][0]
            line_2 = self.tle_dict[name][1]

            tle_slot = TLESlot(self, name, line_1, line_2)
            tle_slot.checkbox.stateChanged.connect(self.get_data_from_enabled_tle_slots)
            tle_slot.checkbox.stateChanged.connect(self.emit_update_signal)

            self.inner_layout.addWidget(tle_slot)

    def get_data_from_enabled_tle_slots(self):
        return_dict = {}

        for i in range(self.inner_layout.count()):
            slot = self.inner_layout.itemAt(i).widget()
            if(isinstance(slot, TLESlot)):
                if slot.checkbox.isChecked():
                    return_dict[slot.tle_name] = [slot.tle_line_1, slot.tle_line_2]
        
        print(f"Return_dict: {return_dict}")
        return return_dict

    def add_elements(self, tle_data):
        print("Adding Elements")
        for i in tle_data:
            name = str(i)
            line_1 = tle_data[name][0]
            line_2 = tle_data[name][1]

            self.tle_dict[name] = [line_1, line_2]

        self.populate_scroll_items()

    def delete_element(self, element_name):
        del self.tle_dict[element_name]

        self.populate_scroll_items()


    def emit_update_signal(self):
        self.update_signal.emit()
        print("Sending update signal")

    def on_file_dropped(self, tle_file_path):
        print("File dropped from: " + tle_file_path)
        tle_data = self.backend.sat_sim_module.handler.read_tle_file(tle_file_path)

        self.add_elements(tle_data)