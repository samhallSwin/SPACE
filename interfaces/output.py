"""
Filename: output.py
Description: Output interface for all component output classes intended to export results to file or next component. 
Author: Elysia Guglielmo
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Implement this interface by assigning this type to a derived class and defining the methods.
Example:
    class SpecialOutput(Output):
        write_to_file(file):
            # functionality here

        set_result():
            # functionality here
"""

import abc

class Output(abc.ABC):

    @abc.abstractmethod
    def write_to_file(file):
        pass

    @abc.abstractmethod
    def set_result():
        pass

    @abc.abstractmethod
    def get_result():
        pass

    @abc.abstractmethod
    def log_result():
        pass
