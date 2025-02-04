import unittest
from multitraitclustering import string_funcs as hp


class TestMultiTraitClustering(unittest.TestCase):

    def test_num_to_word(self):
        """take an integer and convert to TitleCase word."""
        for i in range(-10, 999):
            word = hp.num_to_word(i)
            self.assertTrue(isinstance(word, str))
        self.assertRaises(TypeError, hp.num_to_word, 3.2)
