import unittest
import numpy as np

from random_param_search import random_param_search


class MyTestCase(unittest.TestCase):
    def test_random_param_search(self):
        params_dict = {'learning_rate': {'fn': np.random.uniform, 'kwargs': {'low': 0, 'high': 1}},
                       'n_estimator': {'fn': np.random.randint, 'kwargs': {'low': 10, 'high': 1000}}}
        result = random_param_search(params_dict=params_dict, size=50)

        self.assertEqual(len(result), 5)
        self.assertIsInstance(result[0], dict)


if __name__ == '__main__':
    unittest.main()
