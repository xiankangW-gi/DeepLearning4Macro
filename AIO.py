import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List


class ModelParams:
    """Model parameters and constants"""

    def __init__(self):
        # Core parameters
        self.beta = 0.9
        self.gamma = 2.0

        # State variable parameters
        self.sigma_r = 0.001
        self.rho_r = 0.2
        self.sigma_p = 0.0001
        self.rho_p = 0.999
        self.sigma_q = 0.001
        self.rho_q = 0.9
        self.sigma_delta = 0.001
        self.rho_delta = 0.2
        self.rbar = 1.04

        # Calculate standard deviations for ergodic distributions
        self.sigma_e_r = self.sigma_r / (1 - self.rho_r ** 2) ** 0.5
        self.sigma_e_p = self.sigma_p / (1 - self.rho_p ** 2) ** 0.5
        self.sigma_e_q = self.sigma_q / (1 - self.rho_q ** 2) ** 0.5
        self.sigma_e_delta = self.sigma_delta / (1 - self.rho_delta ** 2) ** 0.5

        # State bounds
        self.w_min = 0.1
        self.w_max = 4.0


class ConsumptionNet(nn.Module):
    """Neural network for consumption decisions"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Initialize weights using He initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class ConsumptionModel:
    """Main model class implementing the consumption-savings problem"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.params = ModelParams()
        self.model = ConsumptionNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.device = device

    @staticmethod
    def min_FB(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - torch.sqrt(a ** 2 + b ** 2)

    def decision_rule(self, r: torch.Tensor, delta: torch.Tensor,
                      q: torch.Tensor, p: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize states
        r = r / (self.params.sigma_e_r * 2)
        delta = delta / (self.params.sigma_e_delta * 2)
        q = q / (self.params.sigma_e_q * 2)
        p = p / (self.params.sigma_e_p * 2)
        w = (w - self.params.w_min) / (self.params.w_max - self.params.w_min) * 2.0 - 1.0

        # Stack inputs
        s = torch.stack([r, delta, q, p, w], dim=1)
        x = self.model(s)

        # Get consumption share and marginal consumption
        zeta = torch.sigmoid(x[:, 0])
        h = torch.exp(x[:, 1])

        return zeta, h

    def residuals(self, e_r: torch.Tensor, e_delta: torch.Tensor, e_q: torch.Tensor, e_p: torch.Tensor,
                  r: torch.Tensor, delta: torch.Tensor, q: torch.Tensor, p: torch.Tensor, w: torch.Tensor):
        # Get current decisions
        zeta, h = self.decision_rule(r, delta, q, p, w)
        c = zeta * w

        # Calculate next period states
        r_next = r * self.params.rho_r + e_r
        delta_next = delta * self.params.rho_delta + e_delta
        p_next = p * self.params.rho_p + e_p
        q_next = q * self.params.rho_q + e_q

        # Calculate next period wealth
        w_next = torch.exp(p_next + q_next) + (w - c) * self.params.rbar * torch.exp(r_next)

        # Get next period decisions
        zeta_next, h_next = self.decision_rule(r_next, delta_next, q_next, p_next, w_next)
        c_next = zeta_next * w_next

        # Calculate residuals
        R1 = (self.params.beta * torch.exp(delta_next - delta) *
              (c_next / c) ** (-self.params.gamma) * self.params.rbar * torch.exp(r_next) - h)
        R2 = self.min_FB(1 - h, 1 - zeta)

        return R1, R2

    def objective(self, n: int) -> torch.Tensor:
        # Generate random states
        r = torch.randn(n, device=self.device) * self.params.sigma_e_r
        delta = torch.randn(n, device=self.device) * self.params.sigma_e_delta
        p = torch.randn(n, device=self.device) * self.params.sigma_e_p
        q = torch.randn(n, device=self.device) * self.params.sigma_e_q
        w = torch.rand(n, device=self.device) * (self.params.w_max - self.params.w_min) + self.params.w_min

        # Generate random shocks
        e1_r = torch.randn(n, device=self.device) * self.params.sigma_r
        e1_delta = torch.randn(n, device=self.device) * self.params.sigma_delta
        e1_p = torch.randn(n, device=self.device) * self.params.sigma_p
        e1_q = torch.randn(n, device=self.device) * self.params.sigma_q

        e2_r = torch.randn(n, device=self.device) * self.params.sigma_r
        e2_delta = torch.randn(n, device=self.device) * self.params.sigma_delta
        e2_p = torch.randn(n, device=self.device) * self.params.sigma_p
        e2_q = torch.randn(n, device=self.device) * self.params.sigma_q

        # Calculate residuals
        R1_e1, R2_e1 = self.residuals(e1_r, e1_delta, e1_p, e1_q, r, delta, q, p, w)
        R1_e2, R2_e2 = self.residuals(e2_r, e2_delta, e2_p, e2_q, r, delta, q, p, w)

        # Calculate squared residuals
        R_squared = R1_e1 * R1_e2 + R2_e1 * R2_e2

        return R_squared.mean()

    def train(self, num_epochs: int, batch_size: int = 1000) -> List[float]:
        losses = []
        pbar = tqdm(range(num_epochs))

        for _ in pbar:
            self.optimizer.zero_grad()
            loss = self.objective(batch_size)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item():.6f}")

        return losses

    def plot_decision_rule(self):
        w_vec = torch.linspace(self.params.w_min, self.params.w_max, 100, device=self.device)
        zeros = torch.zeros_like(w_vec)

        zeta_vec, _ = self.decision_rule(zeros, zeros, zeros, zeros, w_vec)

        plt.figure(figsize=(10, 6))
        plt.title("Multidimensional Consumption-Savings (decision rule)")
        plt.plot(w_vec.cpu(), w_vec.cpu(), linestyle='--', color='black', label='45° line')
        plt.plot(w_vec.cpu(), (w_vec * zeta_vec).cpu(), label='Consumption')
        plt.xlabel("$w_t$")
        plt.ylabel("$c_t$")
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    # Create and train model
    model = ConsumptionModel()
    losses = model.train(num_epochs=50000)

    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(np.sqrt(losses))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot decision rule
    model.plot_decision_rule()


if __name__ == "__main__":
    main()