import torch
import time

from torch import nn


class Teacher:
    def __init__(self, device, model_classifier: nn.Module, model_name: str,
                 batch_size, loss_function, optimizer: torch.optim.Optimizer):

        print("Start initializing Teacher")

        self.batch_size = batch_size

        self.loss_function = loss_function
        self.optimizer = optimizer

        self.device = device
        print(f"Using {self.device} device")

        print("Initializing models")

        self.model = model_classifier.to(self.device)

        print(self.model)
        print("Teacher initialized. Model is ready to be trained")

    def train(self, train_dataloader):
        size = len(train_dataloader)
        print_info_frequency = 100
        self.model.train()

        sum_forward_time = 0
        sum_backprop_time = 0

        current_loss = 0.0

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            forward_start = time.time()
            pred = self.model(X.squeeze())
            forward_time = time.time() - forward_start

            loss = self.loss_function(pred, y)

            backprop_start = time.time()
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            backprop_time = time.time() - backprop_start

            sum_forward_time += forward_time
            sum_backprop_time += backprop_time

            current_loss += loss.item()

            if batch % print_info_frequency == print_info_frequency - 1:
                print(
                    f"avg batch loss: {current_loss / print_info_frequency:>7f}, "
                    f"forward pass time: {forward_time:0.3f} s, "
                    f"backpropagation time: {backprop_time:0.3} s  [{batch:>5d}/{size:>5d}]"
                )

                current_loss = 0.0

    def test(self, test_dataloader):
        size = 0
        num_batches = len(test_dataloader)
        self.model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                # Prepare inputs.
                X, y = X.to(self.device), y.to(self.device)

                # Generate prediction.
                pred = self.model(X.squeeze())

                # Count metrics.
                test_loss += self.loss_function(pred, y).item()
                pred = torch.round(pred.squeeze())

                correct += (pred == y.squeeze()).sum().item()
                size += len(X)

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg batch loss: {test_loss:>8f} \n")
