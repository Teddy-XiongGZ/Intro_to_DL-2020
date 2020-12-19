from matplotlib import pyplot as plt

def plot_loss(losses, title="Loss Curve", label="Train Loss"):
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(losses) + 1), losses, label=label)
    plt.show()

def plot_acc(accs, title="Accuracy", label="Validation Accuracy"):
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.plot(range(1, len(accs) + 1), accs, label=label)
    plt.show()
