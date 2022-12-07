import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               save_path: str,
               model_name: str,
               epoch: int = None,
               optimizer: torch.optim.Optimizer = None,
               loss: float = None) -> None:
    
    save_path = Path(save_path)
    if not save_path.is_dir():
        save_path.mkdir(parents = True,
                        exist_ok = True)
    
    model_save_dir = save_path / model_name.split('/')[0]
    if not model_save_dir.is_dir():
        model_save_dir.mkdir(parents = True,
                             exist_ok = True)
    
    model_save_path = save_path / model_name
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    
    torch.save(obj=checkpoint,
               f=model_save_path)
    print(f'Model saved at {model_save_path}.')
    
def load_model(model: torch.nn.Module,
               load_path: str,
               model_name: str,
               optimizer: torch.optim.Optimizer = None):
    
    load_path = Path(load_path)
    assert load_path.is_dir(), f"Incorrect load path: {load_path}"
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Model name should end with .pt or .pth"
    model_load_path = load_path / model_name
    
    checkpoint = torch.load(model_load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f'Model {model_name} loaded.')
    return model, optimizer, epoch, loss

if __name__ == '__main__':
    from model import AlexNet
    
    model = AlexNet(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch = 1
    loss = 0.1
    
    model_name = 'AlexNet_testsave.pt'
    save_load_path = 'models/'
    
    save_model(model, save_load_path, model_name,
               epoch, optimizer, loss)
    model = AlexNet(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model, optimizer, epoch, loss = load_model(model, save_load_path, model_name,
                                               optimizer)
    print(f'Epoch: {epoch} | loss : {loss}')