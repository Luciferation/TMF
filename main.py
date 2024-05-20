import os
import copy

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import MultiViewDataset
from models.TMF import TMF
from models.TMC import TMC

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def train(model, train_loader, valid_loader, optimizer, step_lr=None, epochs=50, save_weights_to=None, device=get_device()):
    model = model.to(device)

    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, num_samples = 0, 0, 0
        for batch in train_loader:
            x, y, _ = batch['x'], batch['y'], batch['index']
            for k in x.keys():
                x[k] = x[k].to(device)
            y = y.to(device)
            ret = model(x, y, epoch)
            optimizer.zero_grad()
            ret['loss'].mean().backward()
            optimizer.step()

            train_loss += ret['loss'].mean().item() * len(y)
            num_samples += len(y)
            correct += torch.sum(ret['probability'].argmax(dim=-1).eq(y)).item()  # 概率分布得出预测类别

        train_loss = train_loss / num_samples
        train_acc = correct / num_samples

        valid = validate(model, valid_loader)
        if best_valid_acc < valid['accuracy']:
            best_valid_acc = valid['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())

        if step_lr is not None:
            step_lr.step()

        print(f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}', end=' ')
        print(f'train loss {train_loss:6.4f}, train acc {train_acc:6.4f}', end=' ')
        print(f'valid acc {valid["accuracy"]:6.4f}')

    if save_weights_to is not None:
        os.makedirs(os.path.dirname(save_weights_to), exist_ok=True)
        torch.save(best_model_wts, save_weights_to)
    model.load_state_dict(best_model_wts)
    return model


def validate(model, loader, device=get_device()):
    model.eval()
    with torch.no_grad():
        pred = []
        correct, num_samples = 0, 0
        for batch in loader:
            x, y = batch['x'], batch['y']
            for k in x.keys():
                x[k] = x[k].to(device)
            ret = model(x)
            pred.append(ret['probability'].argmax(dim=-1).cpu().numpy())
            correct += torch.sum(ret['probability'].argmax(dim=-1).cpu().eq(y)).item()
            num_samples += len(y)
    pred = np.concatenate(pred)
    acc = correct / num_samples
    return {
        'prediction': pred,
        'accuracy': acc
    }


if __name__ == '__main__':
    train_data = MultiViewDataset(data_path='dataset/handwritten_6views_train.pkl')
    test_data = MultiViewDataset(data_path='dataset/handwritten_6views_test.pkl')
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024)

    model = TMF(
        sample_shapes=[s.shape for s in train_data[0]['x'].values()],
        num_classes=len(set(train_data.y)))
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    print('---------------------------- Experiment ------------------------------')
    print('model:', model.__class__.__name__)
    print('Number of views:', len(train_data.x), 'views with shapes of:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of classes:', len(set(train_data.y)))
    print('Number of train samples:', len(train_data))
    print('Number of test samples:', len(test_data))
    print('Trainable Parameters:')
    for n, p in model.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    model = train(model, train_loader, test_loader, optimizer, step_lr=step_lr, epochs=50)

    valid = validate(model, test_loader)
    print('test predicting accuracy is', valid['accuracy'])
