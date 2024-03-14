import unittest

from hmeg import utils
from hmeg.entities import MinilexPlaceholders


class TestUtils(unittest.TestCase):
    def test_apply_minilex(self):
        with self.subTest("Empty input"):
            self.assertEqual(utils.apply_minilex(""), "")

        with self.subTest("No placeholders"):
            s = "abcde123^%&#$}{"
            self.assertEqual(utils.apply_minilex(s), s)

        with self.subTest("Multiple placeholders"):
            s = "[{verb}][{verb}][{verb}]"
            res = utils.apply_minilex(s)
            with self.assertRaises(ValueError):
                res.index(res[:3], 1)  # no repetitions in the generated string

        with self.subTest("All placeholders"):
            all_placeholders = " ".join(MinilexPlaceholders.to_list())
            res = utils.apply_minilex(all_placeholders)
            for placeholder in all_placeholders:
                self.assertTrue(placeholder not in res)
