import torch 
import tqdm 
from time import default_timer


def train_loop(train_loader, model, optimizer, scheduler, loss_fn, epochs, use_scheduler=True):
    """
    Abstract training loop for a deep learning model.

    Parameters:
    - train_loader: Data loader for the training dataset.
    - model: The neural network model to train.
    - optimizer: The optimization algorithm.
    - scheduler: Learning rate scheduler.
    - loss_fn: Loss function.
    - epochs: Number of training epochs.
    - use_scheduler: Whether to use a learning rate scheduler.

    Returns:
    - Trained model.
    - List of average losses for each epoch.
    - List of times taken for each epoch.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Starting Training on", device)
    model.to(device)

    epoch_losses = []
    epoch_times = []

    tq = tqdm(range(epochs))
    for e in tq:
        losses = []
        start_time = default_timer()
        
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.to("cpu"))

        if use_scheduler:
            scheduler.step()

        end_time = default_timer()
        
        epoch_loss = torch.mean(torch.tensor(losses))
        epoch_losses.append(epoch_loss.item())
        epoch_times.append(end_time - start_time)

        print(f"Epoch {e+1}/{epochs} - Average Epoch loss:", epoch_loss.item())

    return model, epoch_losses, epoch_times