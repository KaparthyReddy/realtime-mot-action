import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, temporal_window=8, image_size=320):
        self.num_samples = num_samples
        self.temporal_window = temporal_window
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'frames': torch.randn(self.temporal_window, 3, self.image_size, self.image_size),
            'boxes': torch.rand(8, 4) * self.image_size,
            'labels': torch.randint(0, 5, (8,)),
            'track_ids': torch.randint(0, 20, (8,)),
            'obj_mask': torch.cat([
                torch.ones(8, dtype=torch.bool),
                torch.zeros(9592, dtype=torch.bool)
            ]),
            'actions': torch.randint(0, 10, (1,))[0]
        }


class TestData(unittest.TestCase):
    """Test data loading and preprocessing"""
    
    def test_dummy_dataset(self):
        """Test dummy dataset creation"""
        dataset = DummyDataset(num_samples=50, temporal_window=8)
        
        self.assertEqual(len(dataset), 50)
        
        sample = dataset[0]
        self.assertIn('frames', sample)
        self.assertIn('boxes', sample)
        self.assertIn('labels', sample)
        self.assertIn('track_ids', sample)
        self.assertIn('actions', sample)
        
        print("✓ DummyDataset test passed")
    
    def test_dataloader(self):
        """Test dataloader"""
        dataset = DummyDataset(num_samples=32)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(loader))
        
        self.assertEqual(batch['frames'].shape[0], 4)
        self.assertEqual(batch['frames'].shape[1], 8)  # temporal window
        self.assertEqual(batch['boxes'].shape[0], 4)
        
        print("✓ DataLoader test passed")
    
    def test_data_shapes(self):
        """Test data shapes are correct"""
        dataset = DummyDataset()
        sample = dataset[0]
        
        # Check shapes
        self.assertEqual(sample['frames'].shape, (8, 3, 320, 320))
        self.assertEqual(sample['boxes'].shape, (8, 4))
        self.assertEqual(sample['labels'].shape, (8,))
        self.assertEqual(sample['track_ids'].shape, (8,))
        
        print("✓ Data shapes test passed")


if __name__ == '__main__':
    unittest.main()