
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import the module by path since it's in examples/
import importlib.util
spec = importlib.util.spec_from_file_location("book_narration_web", str(PROJECT_ROOT / "examples/book_narration_web.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["book_narration_web"] = module
spec.loader.exec_module(module)

split_sentences = module.split_sentences
batch_generator = module.batch_generator

class TestBookNarration(unittest.TestCase):
    def test_split_sentences(self):
        text = "Hello world. This is a test! Is it working? Yes."
        sentences = split_sentences(text)
        expected = ["Hello world.", "This is a test!", "Is it working?", "Yes."]
        self.assertEqual(sentences, expected)
        
        text2 = "No punctuation"
        # Regex expects .!? so this might return the whole text or handle it?
        # Current regex: re.split(r'(?<=[.!?])\s+', text)
        sentences2 = split_sentences(text2)
        self.assertEqual(sentences2, ["No punctuation"])

    def test_batch_generator(self):
        items = [1, 2, 3, 4, 5]
        batches = list(batch_generator(items, 2))
        self.assertEqual(batches, [[1, 2], [3, 4], [5]])

if __name__ == '__main__':
    unittest.main()
