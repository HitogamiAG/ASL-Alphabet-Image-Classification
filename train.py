import argparse
import torch
from torchvision import transforms
from model import AlexNet
from pathlib import Path
import data_setup, engine, utils
import time

parser = argparse.ArgumentParser('parser')
parser.add_argument('-e', '--num_epochs', type=int, default=1)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-m', '--model_name', type=str, default='Unnamed_model' + time.strftime("%Y%m%d_%H%M%S"))
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-f', '--data_folder', type=str, default='data/')
parser.add_argument('-ds', '--data_size', type=str, default='0.9/0.05/0.05')

args = parser.parse_args()

data_path = Path(args.data_folder)
image_path = data_path / 'train'

transform = transforms.Compose([
    transforms.Resize(size=(227, 227)),
    transforms.ToTensor()
])

assert image_path.is_dir(), "Data path doesn't found or not contain train folder"

train_size, val_size, test_size = [float(size) for size in args.data_size.split('/')]

train_loader, val_loader, test_loader, classes = data_setup.load_data(root = image_path,
                 transform=transform,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                batch_size=args.batch_size)

model = AlexNet(intput_shape=3,
                output_shape=len(classes)).to(args.device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

train_results = engine.train(model,
                             args.model_name,
                             train_loader,
                             val_loader,
                             loss_fn=loss_fn,
                             optimizer=optimizer,
                             device=args.device,
                             epochs=args.num_epochs)

