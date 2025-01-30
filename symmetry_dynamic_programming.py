import numpy as np
import quantecon
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置随机种子保证可重复性
torch.manual_seed(123)


# 模型参数配置类
class ModelConfig:
    N = 128  # 企业数量
    T = 63  # 轨迹长度
    num_trajectories = 16

    # 经济模型参数
    alpha_0 = 1.0  # 价格系数
    alpha_1 = 1.0  # 价格系数
    beta = 0.95  # 贴现因子
    gamma = 90.0  # 调整成本系数
    sigma = 0.005  # 个体冲击标准差
    delta = 0.05  # 折旧率
    eta = 0.001  # 总体冲击标准差
    nu = 1.0  # 需求曲率

    # 初始状态参数
    X_0_loc = 0.9
    X_0_scale = 0.05

    # 训练参数
    hidden_dim = 128
    learning_rate = 1e-3
    num_epochs = 21
    batch_size = 16
    print_epoch_frequency = 10
    omega_quadrature_nodes = 7
    L = 4  # 神经网络输出维度


# 神经网络定义
class U_hat_NN(nn.Module):
    def __init__(self, dim_hidden=ModelConfig.hidden_dim):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(ModelConfig.L, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

        self.phi = nn.Sequential(
            nn.Linear(1, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, ModelConfig.L)
        )

    def forward(self, X):
        num_batches, N = X.shape
        phi_X = torch.stack(
            [torch.mean(self.phi(X[i, :].reshape([N, 1])), 0)
             for i in range(num_batches)]
        )
        return self.rho(phi_X)


# 数据生成函数
def generate_initial_data():
    X_0 = torch.normal(ModelConfig.X_0_loc, ModelConfig.X_0_scale,
                       size=(ModelConfig.N,)).abs()
    w = torch.randn(ModelConfig.num_trajectories, ModelConfig.T, ModelConfig.N)
    omega = torch.randn(ModelConfig.num_trajectories, ModelConfig.T, 1)

    def u_0(X):
        return 0.2 - 0.3 * X.mean(1, keepdim=True)

    data = torch.zeros(ModelConfig.num_trajectories, ModelConfig.T + 1, ModelConfig.N)
    data[:, 0, :] = X_0

    for t in range(ModelConfig.T):
        data[:, t + 1, :] = (u_0(data[:, t, :]) +
                             (1 - ModelConfig.delta) * data[:, t, :] +
                             ModelConfig.sigma * w[:, t, :] +
                             ModelConfig.eta * omega[:, t])

    return data.flatten(start_dim=0, end_dim=1)


# 训练设置函数
def setup_training():
    model = U_hat_NN()
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    nodes, weights = quantecon.quad.qnwnorm(ModelConfig.omega_quadrature_nodes)
    quadrature_nodes = torch.tensor(nodes, dtype=torch.float32)
    quadrature_weights = torch.tensor(weights, dtype=torch.float32)

    one_draw_idio_vec = torch.randn(1, ModelConfig.N)
    one_draw_idio = ((one_draw_idio_vec - one_draw_idio_vec.mean()) /
                     one_draw_idio_vec.std())

    return model, optimizer, scheduler, quadrature_nodes, quadrature_weights, one_draw_idio


# 残差计算函数
def compute_residuals(model, batch, quadrature_nodes, quadrature_weights, one_draw_idio):
    X = batch
    u_X = model(X)

    X_primes = torch.stack([
        u_X + (1 - ModelConfig.delta) * X +
        ModelConfig.sigma * one_draw_idio +
        ModelConfig.eta * node
        for node in quadrature_nodes
    ]).type_as(X)

    p_primes = (ModelConfig.alpha_0 -
                ModelConfig.alpha_1 * X_primes.pow(ModelConfig.nu).mean(2))
    Ep = (p_primes.T @ quadrature_weights).type_as(X).reshape(-1, 1)

    Eu = ((torch.stack([model(X_primes[i])
                        for i in range(len(quadrature_nodes))])
           .squeeze(2).T @ quadrature_weights)
          .type_as(X).reshape(-1, 1))

    return (ModelConfig.gamma * u_X -
            ModelConfig.beta * (Ep + ModelConfig.gamma *
                                (1 - ModelConfig.delta) * Eu))


# 模型训练函数
def train_model():
    data = generate_initial_data()
    data_loader = DataLoader(data, batch_size=ModelConfig.batch_size, shuffle=True)
    model, optimizer, scheduler, quadrature_nodes, quadrature_weights, one_draw_idio = setup_training()

    for epoch in range(ModelConfig.num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            residuals = compute_residuals(model, batch, quadrature_nodes,
                                          quadrature_weights, one_draw_idio)
            loss = residuals.pow(2).mean()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % ModelConfig.print_epoch_frequency == 0:
            print(f"epoch = {epoch}, loss = {loss.detach().numpy():.2e}")

    return model


# 新增绘图函数
def plot_results(u_hat_path, u_ref_path, relative_error):
    fontsize = 14
    ticksize = 14
    figsize = (14, 4.5)

    params = {
        'font.family': 'serif',
        'figure.figsize': figsize,
        'figure.dpi': 80,
        'figure.edgecolor': 'k',
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': ticksize,
        'ytick.labelsize': ticksize
    }
    plt.rcParams.update(params)

    plt.subplot(1, 2, 1)
    plt.plot(u_hat_path, color='k', label=r"$u(X_t), \phi$(ReLU)")
    plt.plot(u_ref_path, color='k', linestyle='--', label=r"$u(X_t),$ Linear-Quadratic")
    plt.xlabel(r"Time(t)")
    plt.title(r"$u(X_t)$ with $\phi$(ReLU)")
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(relative_error, color='k')
    plt.xlabel(r"Time(t)")
    plt.title(r"Policy Errors, $\varepsilon_{rel}(X_t)$ with $\phi$(ReLU)")
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    trained_model = train_model()

    # 生成测试数据
    torch.manual_seed(234)
    num_trajectories = ModelConfig.num_trajectories
    T = ModelConfig.T
    N = ModelConfig.N

    w_test = torch.randn(num_trajectories, T, N)
    omega_test = torch.randn(num_trajectories, T, 1)

    # 初始化测试数据集
    X_0 = torch.normal(ModelConfig.X_0_loc, ModelConfig.X_0_scale, size=(N,)).abs()
    data_test = torch.zeros(num_trajectories, T + 1, N)
    data_test[:, 0, :] = X_0

    # 使用训练好的模型生成轨迹
    for t in range(T):
        current_X = data_test[:, t, :]
        with torch.no_grad():
            u_values = trained_model(current_X)
        data_test[:, t + 1, :] = (u_values +
                                  (1 - ModelConfig.delta) * current_X +
                                  ModelConfig.sigma * w_test[:, t, :] +
                                  ModelConfig.eta * omega_test[:, t])

    # 获取第一条测试轨迹
    trajectory_1_test = data_test[0, :, :]
    with torch.no_grad():
        u_hat_path = trained_model(trajectory_1_test).squeeze().numpy()


    # 线性二次型参考解
    def u_ref_lq(X):
        H_0 = 0.06872243906482536
        H_1 = -0.05046135500134341
        return H_0 + H_1 * X.mean(dim=1, keepdim=True)


    # 生成LQ解轨迹
    data_test_lq = torch.zeros_like(data_test)
    data_test_lq[:, 0, :] = X_0
    for t in range(T):
        current_X = data_test_lq[:, t, :]
        u_values = u_ref_lq(current_X)
        data_test_lq[:, t + 1, :] = (u_values +
                                     (1 - ModelConfig.delta) * current_X +
                                     ModelConfig.sigma * w_test[:, t, :] +
                                     ModelConfig.eta * omega_test[:, t])

    # 获取LQ解轨迹
    trajectory_lq = data_test_lq[0, :, :]
    u_ref_path = u_ref_lq(trajectory_lq).squeeze().numpy()

    # 计算相对误差
    relative_error = np.abs((u_hat_path - u_ref_path) / u_ref_path)

    # 调用绘图函数
    plot_results(u_hat_path, u_ref_path, relative_error)