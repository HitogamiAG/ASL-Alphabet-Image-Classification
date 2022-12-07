import torch
from tqdm.auto import tqdm
import utils
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0,0
    
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module,
          model_name: str,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int) -> Dict[str, List[float]]:
    train_results = {
        "train_loss":[],
        "train_acc":[],
        "test_loss":[],
        "test_acc":[]
    }
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}: Train stage')
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        print(f'Epoch {epoch+1}: Test stage')
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        
        train_results["train_loss"].append(train_loss)
        train_results["train_acc"].append(train_acc)
        train_results["test_loss"].append(test_loss)
        train_results["test_acc"].append(test_acc)
        
        print(f'Epoch: {epoch+1} | Train loss: {train_loss} | Train acc: {train_acc}')
        print(f' | Test loss {test_loss} | Test acc: {test_acc}')
        
        utils.save_model(model,
                         'models/',
                         model_name + '/' + f'{model_name}_{epoch+1}.pt',
                         epoch+1,
                         optimizer,
                         test_loss)
        
    return train_results

if __name__ == '__main__':
    from model import AlexNet
    from data_setup import load_data
    from torchvision import transforms
    from pathlib import Path
    
    data_path = Path('data/')
    image_path = data_path / 'train'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    if not image_path.is_dir():
        raise Exception
    
    transform = transforms.Compose([
        transforms.Resize(size=(227, 227)),
        transforms.ToTensor()
    ])
    
    train_dataloader, val_dataloader, _, class_names = load_data(
        root=image_path,
        transform=transform,
        train_size=.1,
        val_size=.1,
        batch_size=32
    )
    
    model = AlexNet(intput_shape=3, output_shape=len(class_names)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=1
    )
    
    print(train_results)