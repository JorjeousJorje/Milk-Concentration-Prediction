import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor

def MSE(pred, truth):
    diff = pred - truth
    N = len(pred)
    # TODO: diff ** 2?
    mse = torch.sum(diff ** 2) / N
    return mse.cpu()

def MAE(pred, truth):
    return torch.abs(pred - truth).mean().cpu()

def RMSE(pred, truth):
    return torch.sqrt(MSE(pred, truth)).cpu()

# def MAE(pred, truth):
#     print(pred, truth)
#     diff = pred - truth
#     N = len(pred)
#     mae = torch.sum(diff) / N
#     return mae.cpu()


def accuracy_metric(predictions, labels):
    preds = torch.argmax(predictions, dim=1)
    correct_samples: float = torch.sum(preds == labels)
    total_samples: float = len(labels)
    
    correct_samples = correct_samples.cpu()
    
    return 100.0 * correct_samples / total_samples


def evaluate_model(model, dataset):
    model.eval()
    
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        signals, labels = next(iter(loader))
        labels = labels.to(model.dummy_param)    
            
        classes, probas = model.predict(signals)
        total = len(dataset)
        correct = (classes == labels).sum()
        accuracy = 100 * correct / total
        return accuracy
        
        
def get_loaders(model_dataset, batch_size: int=100, train_fracture=0.8, val_frac=0.2, test_frac=0.0):
    total_count = len(model_dataset)
    train_count = int(train_fracture * total_count)
    valid_count = int(val_frac * total_count)
    test_count = int(test_frac * total_count)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        model_dataset, (train_count, valid_count, test_count)
    )
    print(test_count)

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_dataset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    dataloaders = {
        "train": train_dataset_loader,
        "val": valid_dataset_loader,
        "test": test_dataset_loader,
    }
    
    datasets = {
        "train": train_dataset,
        "val": valid_dataset,
        "test": test_dataset,
    }
    return dataloaders, datasets
