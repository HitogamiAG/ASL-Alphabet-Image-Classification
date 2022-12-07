import torch
from torch import nn
from torchvision import transforms, models

import data_setup, utils, engine
from model import AlexNet

import argparse
from pathlib import Path

parser = argparse.ArgumentParser('Test script parser')
parser.add_argument('-m', '--model_name', type=str, required=True)
parser.add_argument('-e', '--epoch_num', type=int, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('-f', '--data_folder', type=str, default='data/')
parser.add_argument('-ds', '--data_size', type=str, default='0.9/0.05/0.05')

args = parser.parse_args()

model_path = Path('models/') / args.model_name
assert model_path.is_dir(), f"Can't find model with specified name: {args.model_name}"

model_path = model_path / f'{args.model_name}_{args.epoch_num}.pt'
assert model_path.is_file(), f"Can't find model with specified epoch number: {args.epoch_num}"

data_path = Path(args.data_folder)
image_path = data_path / 'train'

assert image_path.is_dir(), "Data path doesn't found or not contain train folder"

train_size, val_size, test_size = [float(size) for size in args.data_size.split('/')]

weights = models.AlexNet_Weights.IMAGENET1K_V1.DEFAULT
transform = weights.transforms()

train_loader, val_loader, test_loader, classes = data_setup.load_data(root = image_path,
                transform=transform,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                batch_size=args.batch_size)

model = models.alexnet(weights = weights).to(args.device)
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=len(classes))
model.to(args.device)

model, _, _, _ = utils.load_model(model=model,
                                  load_path='models/' + args.model_name + '/',
                                  model_name=f'{args.model_name}_{args.epoch_num}.pt')

loss_fn = torch.nn.CrossEntropyLoss()

y, y_pred, loss, acc = engine.evaluate(model=model,
                                       test_dataloader=test_loader,
                                       loss_fn=loss_fn,
                                       device=args.device)

print(f'Evaluated loss {round(loss, 3)} and accuracy {round(acc, 3)} on {len(test_loader)} images')

class_to_idx = {classes[i]:i for i in range(len(classes))}

utils.plot_confusion_matrix(y_pred.to('cpu'), y.to('cpu'), class_to_idx)
utils.plot_classification_report(y_pred.to('cpu'), y.to('cpu'), class_to_idx)