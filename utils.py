import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
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
    
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print('Optimizer can"t be loaded')
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f'Model {model_name} loaded.')
    return model, optimizer, epoch, loss

def plot_confusion_matrix(y_pred: torch.Tensor,
                          y: torch.Tensor,
                          class_to_idx: Dict[str, int]):
    ConfMat = MulticlassConfusionMatrix(len(class_to_idx))
    conf_mat = ConfMat(y_pred, y)
    sns.heatmap(conf_mat, annot_kws=class_to_idx)
    plt.title('Confusion matrix')
    plt.show()
    
def plot_classification_report(y_pred: torch.Tensor,
                          y: torch.Tensor,
                          class_to_idx: Dict[str, int]):
    precision = MulticlassPrecision(num_classes=len(class_to_idx), average=None)
    recall = MulticlassRecall(num_classes=len(class_to_idx), average=None)
    f1_score = MulticlassF1Score(num_classes=len(class_to_idx), average=None)
    
    precision_results = precision(y_pred, y).tolist()
    recall_results = recall(y_pred, y).tolist()
    f1_score_results = f1_score(y_pred, y).tolist()

    scores = [precision_results, recall_results, f1_score_results]
    scores = np.array(scores).T
    print(pd.DataFrame(data=scores, columns=['Precision', 'Recall', 'F1-Score'], index=class_to_idx.keys()).to_markdown())

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