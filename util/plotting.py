import ast
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        data_dict = ast.literal_eval(data)
    return data_dict

def plot_on_axis(ax, data, title, ylabel, ydata_key, label, ydata_transform=lambda x: x):
    ax.plot([ydata_transform(y) for y in data[ydata_key]], label=label)
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def plot_train_loss(ax, data, label):
    plot_on_axis(ax, data, "Training Loss", "Loss", 'train_losses_history', label)

def plot_train_accuracy(ax, data, label):
    plot_on_axis(ax, data, "Training Accuracy", "Accuracy", 'train_corrects_history', label, float)

def plot_val_loss(ax, data, label):
    plot_on_axis(ax, data, "Validation Loss", "Loss", 'val_losses_history', label)

def plot_val_accuracy(ax, data, label):
    plot_on_axis(ax, data, "Validation Accuracy", "Accuracy", 'val_corrects_history', label, float)

def plot_epoch_times(ax, data, label):
    plot_on_axis(ax, data, "Epoch Times", "Time (seconds)", 'epoch_times_history', label)


def plot_all_metrics(data_list, title="Model Performance"):
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))  

    for i, data in enumerate(data_list):
        label = f'Dataset {i+1}'
        plot_train_loss(axs[0], data, label)
        plot_train_accuracy(axs[1], data, label)
        plot_val_loss(axs[2], data, label)
        plot_val_accuracy(axs[3], data, label)
        plot_epoch_times(axs[4], data, label)

    plt.tight_layout()
    plt.show()

