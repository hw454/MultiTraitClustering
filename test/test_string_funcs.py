"""
Author: Hayley Wragg
Description: TestStringFuncs unit tests for the functions in the string_funcs module
Type: UnitTests
Created: 5th February 2025
"""
import unittest
from multitraitclustering import string_funcs as hp


class TestStringFuncs(unittest.TestCase):
    """
    TestStringFuncs unit tests for the functions in the string_funcs module

    :param unittest: unittest.TestCase
    """

    def test_num_to_word(self):
        """take an integer and convert to TitleCase word."""
        for i in range(-10, 999):
            word = hp.num_to_word(i)
            self.assertTrue(isinstance(word, str))
        self.assertRaises(TypeError, hp.num_to_word, 3.2)

if __name__ == '__main__':
    unittest.main()
