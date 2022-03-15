from tqdm import tqdm
import torch
from dataset import TrainDataset
from torch.utils.data import DataLoader
from net import Net
from torch import optim
from loss_function import Loss
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt


def train_loop(dataloader, model, loss_func, optimizer, device):
    batches = tqdm(dataloader)
    total_loss = 0.0
    for inputs, targets in batches:
        predictions = model(inputs.to(device))
        loss = loss_func(predictions, targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cpu = loss.cpu().item()
        batches.set_postfix(loss=loss_cpu)
        total_loss += loss_cpu

    ave_loss = (total_loss/len(batches))
    return ave_loss


def save_result(model, losses):
    model_dir = os.path.join('model', datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)

    losses_data = pd.DataFrame({'epoch': range(len(losses)), 'loss': losses})
    losses_data.to_csv(os.path.join(model_dir, 'losses.csv'))

    plt.figure()
    plt.plot(losses)
    plt.savefig(os.path.join(model_dir, 'losses.png'))


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'start training on {device} ...')

    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = Net().to(device)
    loss_func = Loss(7, 2, 5, 0.5, device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    losses = []
    epochs = 100
    for i in range(epochs):
        print(f'start epoch {i} ----------------------')
        loss = train_loop(train_dataloader, model, loss_func, optimizer, device)

        loss = 0
        losses.append(loss)

    save_result(model, losses)
    print('done')
