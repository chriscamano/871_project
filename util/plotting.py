import ast
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        data_dict = ast.literal_eval(data)
    return data_dict


def plot_train_loss(data, title="Training Loss"):
    plt.figure()
    plt.plot(data['train_losses_history'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_train_accuracy(data, title="Training Accuracy"):
    plt.figure()
    train_acc = [float(acc) for acc in data['train_corrects_history']]
    plt.plot(train_acc)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def plot_val_loss(data, title="Validation Loss"):
    plt.figure()
    val_loss = [loss for loss in data['val_losses_history']]
    plt.plot(val_loss)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_val_accuracy(data, title="Validation Accuracy"):
    plt.figure()
    val_acc = [float(acc) for acc in data['val_corrects_history']]
    plt.plot(val_acc)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def plot_epoch_times(data, title="Epoch Times"):
    plt.figure()
    plt.plot(data['epoch_times_history'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.show()

def plot_all_metrics(data, title="Model Performance"):
    fig, axs = plt.subplots(5, 1, figsize=(10, 20))  # Adjust figsize as needed

    # Training Loss
    axs[0].plot(data['train_losses_history'])
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Training Accuracy
    train_acc = [float(acc) for acc in data['train_corrects_history']]
    axs[1].plot(train_acc)
    axs[1].set_title('Training Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)

    # Validation Loss
    val_loss = [loss for loss in data['val_losses_history']]
    axs[2].plot(val_loss)
    axs[2].set_title('Validation Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    # Validation Accuracy
    val_acc = [float(acc) for acc in data['val_corrects_history']]
    axs[3].plot(val_acc)
    axs[3].set_title('Validation Accuracy')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Accuracy')
    axs[3].grid(True)

    # Epoch Times
    axs[4].plot(data['epoch_times_history'])
    axs[4].set_title('Epoch Times')
    axs[4].set_xlabel('Epochs')
    axs[4].set_ylabel('Time (seconds)')
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()



