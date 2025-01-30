import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm


class ModelParameters:
    # Economic parameters
    alpha = 1.0 / 3.0
    beta = 0.9
    delta = 0.1

    # Grid parameters
    k_min = 0.4
    k_max = 4.0
    grid_size = 16

    # Training parameters
    hidden_dim = 128
    learning_rate = 1e-3
    num_epochs = 2001
    print_frequency = 200

    # Testing parameters
    test_periods = 50
    initial_capital = 0.5


class EconomicFunctions:
    @staticmethod
    def production(k):
        return k ** ModelParameters.alpha

    @staticmethod
    def marginal_product(k):
        return ModelParameters.alpha * (k ** (ModelParameters.alpha - 1))

    @staticmethod
    def marginal_utility(c):
        return c ** (-1)

    @staticmethod
    def steady_state():
        base = ((1.0 / ModelParameters.beta) - 1.0 + ModelParameters.delta) / ModelParameters.alpha
        k_star = base ** (1.0 / (ModelParameters.alpha - 1))
        c_star = EconomicFunctions.production(k_star) - ModelParameters.delta * k_star
        return k_star, c_star


class CapitalModel(nn.Module):
    def __init__(self, dim_hidden=ModelParameters.hidden_dim):
        super().__init__()
        self.k_prime = nn.Sequential(
            nn.Linear(1, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1),
            nn.Softplus(beta=1.0)
        )

    def forward(self, x):
        return self.k_prime(x)


def setup_training():
    torch.manual_seed(123)

    # Create capital grid
    capital_grid = torch.linspace(ModelParameters.k_min,
                                  ModelParameters.k_max,
                                  ModelParameters.grid_size).unsqueeze(dim=1)
    data_loader = DataLoader(capital_grid, batch_size=len(capital_grid), shuffle=False)

    # Initialize model and optimizer
    model = CapitalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelParameters.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

    return model, optimizer, scheduler, data_loader


def compute_residuals(model, batch):
    k_t = batch
    k_tp1 = model(k_t)
    k_tp2 = model(k_tp1)

    c_t = EconomicFunctions.production(k_t) + (1 - ModelParameters.delta) * k_t - k_tp1
    c_tp1 = EconomicFunctions.production(k_tp1) + (1 - ModelParameters.delta) * k_tp1 - k_tp2

    return (EconomicFunctions.marginal_utility(c_t) /
            EconomicFunctions.marginal_utility(c_tp1) -
            ModelParameters.beta * (1 - ModelParameters.delta +
                                    EconomicFunctions.marginal_product(k_tp1)))


def train_model(model, optimizer, scheduler, data_loader):
    for epoch in range(ModelParameters.num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            residuals = compute_residuals(model, batch)
            loss = residuals.pow(2).mean()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % ModelParameters.print_frequency == 0:
            print(f"epoch = {epoch}, loss = {loss.detach().numpy():.2e}")

    return model


def generate_paths(model):
    T = ModelParameters.test_periods
    k_path = torch.empty(T + 1).unsqueeze(1)
    c_path = torch.empty(T).unsqueeze(1)
    time_grid = torch.arange(T).unsqueeze(1)

    k_path[0] = ModelParameters.initial_capital
    for t in range(T):
        k_path[t + 1] = model(k_path[t])
        c_path[t] = (EconomicFunctions.production(k_path[t]) +
                     (1 - ModelParameters.delta) * k_path[t] - k_path[t + 1])

    return k_path.detach(), c_path.detach(), time_grid.detach()


def plot_results(k_path, c_path, time_grid, residuals):
    k_star, c_star = EconomicFunctions.steady_state()

    plt.style.use('seaborn-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Capital path
    axes[0].plot(time_grid, k_path[:-1], color='k', label=r"$\hat{k}(t)$")
    axes[0].axhline(y=k_star, linestyle='--', color='k', label="Steady-State")
    axes[0].set_xlabel("Time(t)")
    axes[0].set_title("Capital Path")
    axes[0].set_ylim([k_path[0].item() - 0.1, k_star * (1 + 0.35)])
    axes[0].legend()

    # Consumption path
    axes[1].plot(time_grid, c_path, color='b', label=r"$\hat{c}(t)$")
    axes[1].axhline(y=c_star, linestyle='--', color='b', label="Steady-State")
    axes[1].set_xlabel("Time(t)")
    axes[1].set_title("Consumption Path")
    axes[1].set_ylim([c_path[0].item() - 0.1, c_star * (1 + 0.25)])
    axes[1].legend()

    # Residuals
    axes[2].plot(time_grid, residuals ** 2, color='k', label="Squared Residuals")
    axes[2].set_xlabel("Time(t)")
    axes[2].set_title("Euler Residuals Squared")
    axes[2].set_yscale('log')
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def main():
    # Setup and training
    model, optimizer, scheduler, data_loader = setup_training()
    trained_model = train_model(model, optimizer, scheduler, data_loader)

    # Generate and plot results
    k_path, c_path, time_grid = generate_paths(trained_model)
    residuals = compute_residuals(trained_model, k_path[:-1])
    plot_results(k_path, c_path, time_grid, residuals.detach())


if __name__ == "__main__":
    main()