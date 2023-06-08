import torch
import torch.nn as nn
import torch.optim as optim
from translation_dataset import get_english_german_translation_dataset, split_dataset
from torch.utils.data import DataLoader
from model import Transformer
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str, log_dir: str):
        """
        Trainer class for training and evaluation of a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (str): Device to use for training ("cpu" or "cuda").
            log_dir (str): Directory path for logging with TensorBoard.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.metrics = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

        self.epoch = 0
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self):
        """
        Performs a single training epoch.
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= len(self.train_loader)
        train_accuracy = 100.0 * correct / total

        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_accuracy'].append(train_accuracy)

        self.writer.add_scalar('Loss/Train', train_loss, self.epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, self.epoch)

        print(f"Epoch: {self.epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    def validate_epoch(self):
        """
        Performs a single validation epoch.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(self.val_loader)
        val_accuracy = 100.0 * correct / total

        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_accuracy'].append(val_accuracy)

        self.writer.add_scalar('Loss/Validation', val_loss, self.epoch)
        self.writer.add_scalar('Accuracy/Validation', val_accuracy, self.epoch)

        print(f"Epoch: {self.epoch} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    def train(self, num_epochs: int):
        """
        Trains the model for the specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            self.train_epoch()
            self.validate_epoch()
            self.epoch += 1

        print("Training complete!")

    def save_checkpoint(self, filepath: str):
        """
        Saves the current model checkpoint.

        Args:
            filepath (str): File path to save the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def close_writer(self):
        """
        Closes the TensorBoard SummaryWriter.
        """
        self.writer.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    context_size = 64
    test_size = .2

    dataset = get_english_german_translation_dataset(context_size=context_size)
    train_dataset, test_dataset = split_dataset(dataset=dataset, test_size=test_size)

    # Create a data loader for batching
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    transformer = Transformer(source_vocab_size=dataset.source_vocab_size,
                              target_vocab_size=dataset.target_vocab_size,
                              embedding_size=16, context_size=context_size)

    log_dir = "logs"
    trainer = Trainer(model=transformer, train_loader=train_loader, val_loader=val_loader, device=device,
                      log_dir=log_dir)
    trainer.train(1000)
    trainer.close_writer()
