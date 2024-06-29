"""
Filename: config.py
Description: Config interface for all component config classes intended to process simulation options. 
Author: Elysia Guglielmo
Date: 2024-06-29
Version: 1.0
Python Version: 

Changelog:
- 2024-06-29: Initial creation.

Usage: 
Implement this interface by assigning this type to a derived class and defining the methods.
Example:
    class SpecialConfig(Config):
        read_options(json):
            # functionality here

        read_options_from_file(file):
            # functionality here
"""

import abc

class Config(abc.ABC):

    @abc.abstractmethod
    def read_options(json):
        pass

    @abc.abstractmethod
    def read_options_from_file(file):
        pass