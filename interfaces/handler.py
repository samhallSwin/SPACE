"""
Filename: handler.py
Description: Handler interface for all component input classes intended to parse data from an expected format. 
Author: Elysia Guglielmo
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Implement this interface by assigning this type to a derived class and defining parse_input() method.
Example:
    class SpecialHandler(Handler)
        parse_input(file):
            # functionality here
"""

import abc

class Handler(abc.ABC):

    @abc.abstractmethod
    def parse_file(file):
        pass

    @abc.abstractmethod
    def parse_data(data):
        pass