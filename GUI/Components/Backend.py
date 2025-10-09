import sys

# setting path
sys.path.append('../SPACE')

import module_factory
from module_factory import ModuleKey

from workflows import flomps

import json

class Backend():

    def __init__(self):
        print("Backend - Init Started")

        self._subscribers = []

        self.tle_dict = {}
        self.tle_status = {}

        options = self.read_options_file()
        self.sat_sim_module = module_factory.create_sat_sim_module()
        self.sat_sim_module.config.read_options(options["sat_sim"])

        self.instance = self.sat_sim_module.handler.sat_sim
        
        print("Backend - Init complete")

    #Yoinked from main.py
    def read_options_file(self):
        """Parse the current JSON file in use for options configuration."""
        file = self.get_config_file()
        with open(file) as f:
            return json.load(f)

    def get_config_file(self):
        """Get the current JSON file in use for options configuration."""
        config_file = ''
        with open('settings.json') as f:
            options = json.load(f)
            config_file = options['config_file']

        return config_file
    
    def get_data_from_enabled_tle_slots(self):
        return_dict = None

        for i in self.tle_dict:
            if(self.tle_status[i]):
                if return_dict is None:
                    return_dict = {}

                name = str(i)
                line_1 = self.tle_dict[name][0]
                line_2 = self.tle_dict[name][1]
                
                return_dict[name] = [line_1, line_2]

        return return_dict

    #TODO: add functionality to manipulate tle_dict (Add, Remove, RemoveAt)
    def add_elements(self, tle_data):
        for i in tle_data:
            name = str(i)
            line_1 = tle_data[name][0]
            line_2 = tle_data[name][1]

            self.add(name, line_1, line_2)
            

        self._notify()
        
    def add(self, name, line_1, line_2):
        self.tle_dict[name] = [line_1, line_2]
        self.tle_status[name] = True
    #TODO: link tle_status[name] to the checkbox state, rather than hardcoding it 

    def delete(self, element_name):
        del self.tle_dict[element_name]
        del self.tle_status[element_name]

        self._notify()
    
    def update_all_element_states(self, element_states):
        for i in element_states:
            if self.tle_status[str(i)] != i:
                self.set_element_state(str(i), element_states[i])

        self._notify()

    def set_element_state(self, element_name, element_state):

        self.tle_status[element_name] =  element_state


    def subscribe(self, callback):
        """Register a function to be called on update."""
        self._subscribers.append(callback)

    def _notify(self):
        """Call all subscribed functions."""
        for callback in self._subscribers:
            callback()