"""
Filename: unit_test_algorithm_component.py
Description: Script to execute unit test case for algorithm component.
Author: Yuganya Perumal
Date: 2024-09-27
Version: 1.0
Python Version: 3.12.3

Usage: 
Unit test case scripts for Algorithm Component.
"""
import unittest
import numpy as np
import pandas as pd
from algorithm_core import Algorithm
from algorithm_handler import AlgorithmHandler
from algorithm_output import AlgorithmOutput

class TestAlgorithmCore(unittest.TestCase):
    def setUp(self):
        # Set up test objects necessary for algorithm core execution.
        self.algorithm = Algorithm()
        self.adjacency_matrix = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]])
        self.algorithm.set_satellite_names(["Satellite 1", "Satellite 2", "Satellite 3", "Satellite 4"])
        self.algorithm.set_adjacency_matrices([("2024-10-04 13:23:24", self.adjacency_matrix)])
    
    def test_set_get_satellite_names(self):
        # Test setter and getters satellite names in Algorithm Core.
        self.algorithm.set_satellite_names(["NovaSAR-1", "NovaSAR-2", "NovaSAR-3"])
        self.assertEqual(self.algorithm.get_satellite_names(), ["NovaSAR-1", "NovaSAR-2", "NovaSAR-3"])

    def test_set_get_adjacency_matrices(self):
        # Test setters and getters of adjacency matrices in Algorithm Core.
        matrices = [("2024-10-04 15:40:00", np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]))]
        self.algorithm.set_adjacency_matrices(matrices)
        self.assertEqual(self.algorithm.get_adjacency_matrices(), matrices)
    
    def test_select_satellite_with_max_connections(self):
        # Test selection of satellite with maximum connections
        selected_index, max_connections = self.algorithm.select_satellite_with_max_connections(self.adjacency_matrix)
        self.assertEqual(selected_index, 0)
        self.assertEqual(max_connections, 3)
    
    def test_select_satellite_with_previous_selection_count(self):
        # Create two adjacency matrices to simulate satellite selection
        adjacency_matrix_1 = np.array([
            [0, 1, 1, 1],  
            [1, 0, 0, 0],  
            [1, 0, 0, 1],  
            [1, 0, 1, 0]   
        ])
        
        adjacency_matrix_2 = np.array([
            [0, 0, 1, 1],  
            [0, 0, 0, 0],  
            [1, 0, 0, 1], 
            [1, 0, 1, 0]   
        ])
        self.algorithm.set_satellite_names(["Satellite 1", "Satellite 2", "Satellite 3", "Satellite 4"])
        self.algorithm.set_adjacency_matrices([("2024-10-04 10:00:00", adjacency_matrix_1), ("2024-10-04 11:00:00", adjacency_matrix_2)])
        self.algorithm.selection_counts = np.array([0, 0, 0, 0])
        
        # Step 1: Select satellite based on the first matrix
        selected_index_1, max_connections_1 = self.algorithm.select_satellite_with_max_connections(adjacency_matrix_1)
        
        # Assert that it selects the satellite with the highest connections in the first matrix (Satellite 1)
        self.assertEqual(selected_index_1, 0)
        self.assertEqual(max_connections_1, 3)
        
        # Step 2: Select satellite based on the second matrix
        selected_index_2, max_connections_2 = self.algorithm.select_satellite_with_max_connections(adjacency_matrix_2)
        
        # Since Satellite 1 was selected before, it should now prefer a satellite with fewer selections.
        self.assertEqual(selected_index_2, 2)  # Assuming satellite 3 now has the fewest selections with max connections in matrix 2
        self.assertEqual(max_connections_2, 2)

    
    def test_get_selected_satellite_name(self):
        # Test getting selected satellite name
        satellite_name = self.algorithm.get_selected_satellite_name(1)
        self.assertEqual(satellite_name, "Satellite 2")
    
    def test_start_algorithm_steps(self):
        # Test algorithm component's steps execution for satellite count greater than 1.
        self.algorithm.start_algorithm_steps()
        result = self.algorithm.output.get_result()
        self.assertIsNotNone(result)
        self.assertIn("2024-10-04 13:23:24", result['time_stamp'].values)
        self.assertIn(4, result['satellite_count'].values)
        self.assertIn("Satellite 1", result['satellite_name'].values)
        self.assertIn(True, result['aggregator_flag'].values)
        self.assertTrue(any(np.array_equal(self.adjacency_matrix, matrix) for matrix in result['federatedlearning_adjacencymatrix'].values))

    def test_start_algorithm_steps_one_satellite(self):
        # Test algorithm component's steps execution for satellite count equal to 1.
        self.adjacency_matrix = np.array([[0]])
        self.algorithm.set_satellite_names(["Satellite 1"])
        self.algorithm.set_adjacency_matrices([("2024-10-04 14:23:24", self.adjacency_matrix)])
        self.algorithm.start_algorithm_steps()
        result = self.algorithm.output.get_result()
        self.assertIsNotNone(result)
        self.assertIn("2024-10-04 14:23:24", result['time_stamp'].values)
        self.assertIn(1, result['satellite_count'].values)
        self.assertIn(None, result['satellite_name'].values)
        self.assertIn("None", result['aggregator_flag'].values)
        self.assertTrue(any(np.array_equal(self.adjacency_matrix, matrix) for matrix in result['federatedlearning_adjacencymatrix'].values))

class MockAlgorithm:
    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = None
        self.satellite_names = None

    def set_adjacency_matrices(self, adj_matrices):
        self.adjacency_matrices = adj_matrices

    def set_satellite_names(self, sat_names):
        self.satellite_names = sat_names
class TestAlgorithmHandler(unittest.TestCase):

    def setUp(self):
        # Set up test objects necessary for algorithm handler execution.
        self.algorithm = Algorithm()
        self.handler = AlgorithmHandler(self.algorithm)
        self.mock_algorithm = MockAlgorithm()
        self.mock = AlgorithmHandler(self.mock_algorithm)
    
    def test_read_adjacency_matrices_invalid_file(self):
        # Test reading from a non-existing file
        with self.assertRaises(FileNotFoundError):
            self.handler.read_adjacency_matrices("non_existing_file.txt")
    
    def test_validate_adjacency_matrices(self):
        # Test validation of adjacency matrices properties of square and symmetry.
        matrices = [("2024-10-04 15:40:00", np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]))]
        self.assertTrue(self.handler.validate_adjacency_matrices(matrices))
    
    def test_validate_adjacency_matrices_not_symmetry(self):
        # Test validation of adjacency matrices properties of square and symmetry.
        matrices = [("2024-10-04 15:40:00", np.array([[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]))]
        with self.assertRaises(ValueError) as context:
            self.handler.validate_adjacency_matrices(matrices)
        self.assertIn("is not symmetric", str(context.exception))

    def test_validate_adjacency_matrices_not_square(self):
        # Test validation of adjacency matrices properties of square and symmetry.
        matrices = [("2024-10-04 15:40:00", np.array([[0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]]))]
        with self.assertRaises(ValueError) as context:
            self.handler.validate_adjacency_matrices(matrices)
        self.assertIn("is not square", str(context.exception))
    
    def test_auto_generate_satellite_names(self):
        # Test automatic generation of satellite names
        self.handler.auto_generate_satellite_names(3)
        self.assertEqual(self.handler.sat_names, ["Satellite 1", "Satellite 2", "Satellite 3"])
    
    def test_parse_data(self):
        # Test parsing of data
        matrices = [("2024-10-04 15:40:00", np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]))]
        self.handler.parse_data(matrices)
        self.assertEqual(self.algorithm.get_satellite_names(), ["Satellite 1", "Satellite 2", "Satellite 3", "Satellite 4"])
        self.assertFalse(any(np.array_equal(self.algorithm.get_adjacency_matrices(), matrix) for matrix in matrices))
 
    def test_parse_file(self):
        # Mock the file read method to avoid file I/O in the test
        def mock_read_adjacency_matrices(file_name):
            return [("2024-10-04 15:40:00", np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]))]

        self.mock.read_adjacency_matrices = mock_read_adjacency_matrices
        self.mock.parse_file("mock_file.txt")
        self.assertIsNotNone(self.mock.adjacency_matrices)
        self.assertEqual(len(self.mock.adjacency_matrices), 1)
        self.assertTrue(np.array_equal(self.mock.adjacency_matrices[0][1], np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]])))
        self.assertEqual(self.mock_algorithm.satellite_names, ["Satellite 1", "Satellite 2", "Satellite 3", "Satellite 4"])
class TestAlgorithmOutput(unittest.TestCase):

    def setUp(self):
        # Set up test objects necessary for algorithm output execution.
        self.output = AlgorithmOutput()
        self.algorithm_output = {
            "2024-10-04 13:23:24": {
                'satellite_count': 4,
                'selected_satellite': "Satellite 1",
                'federatedlearning_adjacencymatrix': np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]),
                'aggregator_flag': True
            }
        }
    def test_process_algorithm_output(self):
        # Test algorithm output can be processed and provide a data frame data structure
        
        self.output.process_algorithm_output(self.algorithm_output)
        result_df = self.output.get_flam()
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]['satellite_name'], "Satellite 1")
    
    def test_write_to_file(self):
        # Test writing algorithm output to a file
        self.output.write_to_file(self.algorithm_output)
        with open('Federated_Learning_Adjacency_Matrix.txt', 'r') as file:
            content = file.read()
            self.assertIn('Satellite 1', content)
    
    def test_set_and_get_result(self):
        # Test algorithm output can set in result for FL component to consume.
        self.output.set_result(self.algorithm_output)
        result = self.output.get_result()
        expected_output = {
                'time_stamp':["2024-10-04 13:23:24"],
                'satellite_count': [4],
                'satellite_name': ["Satellite 1"],
                'aggregator_flag': [True],   
                'federatedlearning_adjacencymatrix': [np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]])]              
        }
        expected_df = pd.DataFrame(expected_output)
        pd.testing.assert_frame_equal(result, expected_df, check_dtype=False)

if __name__ == '__main__':
    with open('test_algorithm_component_report.txt', 'w') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner)

