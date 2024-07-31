"""
Filename: input.py
Description: Input interface for all component input classes intended to parse data from an expected format. 
Author: Elysia Guglielmo
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Implement this interface by assigning this type to a derived class and defining parse_input() method.
Example:
    class SpecialInput(Input)
        parse_input(file):
            # functionality here
"""

import abc

class Input(abc.ABC):

    @abc.abstractmethod
    def parse_input(file):
        pass