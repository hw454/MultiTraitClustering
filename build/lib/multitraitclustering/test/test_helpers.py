import unittest
import helpers as hp

class TestMultiTraitClustering(unittest.TestCase):

    def test_num_to_word(self):
        """The function num_to_word should take an integer and convert it into a TitleCase word.
        """
        for i in range(-10,999):
            word = hp.num_to_word(i)
            self.assertTrue(isinstance(word, str))
        self.assertRaises(TypeError, hp.num_to_word, 3.2)