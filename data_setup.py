import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from pathlib import Path
import numpy as np

from os import cpu_count


def load_data(root: str,
              transform: transforms.Compose,
              train_size: float = 0.8,
              val_size: float = None,
              test_size: float = None,
              shuffle: bool = True,
              batch_size: int = 1,
              num_workers: int = cpu_count(),
              seed: int = 42):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = datasets.ImageFolder(root=root,
                                   transform=transform,
                                   target_transform=None)
    
    print(f'Loaded {len(dataset)} images.')
    
    class_names = dataset.classes
    print(f'Loaded {len(class_names)} classes.')
    
    val_size = val_size if val_size is not None else 1 - train_size - test_size if test_size is not None else (1 - train_size) / 2
    test_size = test_size if test_size is not None else 1 - train_size - val_size
    
    val_size, test_size = np.round(val_size, 2), np.round(test_size, 2)
    
    print(f'Train / val / test size: {train_size} / {val_size} / {test_size} ratio')
    
    train_size = int(train_size * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = int(test_size * len(dataset))
    
    print(f'Train / val / test size: {train_size} / {val_size} / {test_size} images')
        
    indicies = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indicies)
    train_indicies, val_indicies, test_indicies = (indicies[:train_size],
                                                   indicies[train_size:train_size+val_size],
                                                   indicies[train_size+val_size:])
    train_sampler = SubsetRandomSampler(train_indicies)
    val_sampler = SubsetRandomSampler(val_indicies)
    test_sampler = SubsetRandomSampler(test_indicies)
    
    train_loader = DataLoader(dataset=dataset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=num_workers)
    print('Train data is loaded')
    val_loader = DataLoader(dataset=dataset,
                              sampler=val_sampler,
                              batch_size=batch_size,
                              num_workers=num_workers)
    print('Validation data is loaded')
    test_loader = DataLoader(dataset=dataset,
                              sampler=test_sampler,
                              batch_size=batch_size,
                              num_workers=num_workers)
    print('Test data is loaded')
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == '__main__':
    data_path = Path('data/')
    image_path = data_path / 'train'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if image_path.is_dir():
        train_loader, _, _, classes = load_data(root = image_path,
                  transform=transform,
                  train_size=0.9,
                  val_size=0.05,
                  test_size=0.05,
                  batch_size=32)
        
    print(next(iter(train_loader))[0].shape)
    print(classes)
