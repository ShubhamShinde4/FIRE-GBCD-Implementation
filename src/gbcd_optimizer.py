# src/gbcd_optimizer.py

import torch

class GBCDOptimizer:
    def __init__(self, sparsity_penalty=0.1, fusion_penalty=0.1, device=None, learning_rate=0.01):
        self.sparsity_penalty = sparsity_penalty
        self.fusion_penalty = fusion_penalty
        self.learning_rate = learning_rate
        # Automatically detect GPU if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def optimize(self, rule_weights, max_iter=100):
        # Convert rule weights to GPU tensor
        weights = torch.tensor(rule_weights, device=self.device, dtype=torch.float32, requires_grad=True)

        # Optimization loop
        for iteration in range(max_iter):
            # Compute loss
            loss = self._objective_function(weights)
            
            # Compute gradients using autograd
            loss.backward()

            # Update rule weights manually using gradients
            with torch.no_grad():
                weights -= self.learning_rate * weights.grad  # Gradient descent step
                weights.grad.zero_()  # Reset gradients after each iteration

            # Optional: Print loss every 10 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss.item()}")

        # Convert the tensor back to a numpy array and return
        return weights.cpu().detach().numpy()

    def _objective_function(self, weights):
        """
        Objective function for GBCD:
        - Sparsity-inducing penalty (encourages zero values)
        - Fusion penalty (encourages similar rules to merge)
        """
        mse_loss = torch.mean((weights - torch.mean(weights)) ** 2)
        sparsity_loss = self.sparsity_penalty * torch.sum(torch.abs(weights))
        fusion_loss = self.fusion_penalty * torch.sum((weights[1:] - weights[:-1]) ** 2)

        # Total loss
        total_loss = mse_loss + sparsity_loss + fusion_loss
        return total_loss
