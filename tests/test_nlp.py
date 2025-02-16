import unittest
import torch
from nlp_model.train_bert import ChemistryDataset

class TestNLP(unittest.TestCase):
    def test_dataset(self):
        dummy_encodings = {
            'input_ids': torch.randint(0, 1000, (10, 128)),
            'attention_mask': torch.ones((10, 128))
        }
        dummy_labels = [0]*5 + [1]*5
        dataset = ChemistryDataset(dummy_encodings, dummy_labels)
        self.assertEqual(len(dataset), 10)
        sample = dataset[0]
        self.assertIn('input_ids', sample)