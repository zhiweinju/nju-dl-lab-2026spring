# 实验任务二：Actor-Critic

!!! abstract "实验目标"
    通过本次实验，你将掌握以下内容：

    1. 理解基于策略的方法（policy-based）与基于价值的方法（value-based）的区别。
    2. 掌握 Actor-Critic 框架中 Actor（策略网络）与 Critic（价值网络）的结构与协作方式。
    3. 实现 TD 误差驱动的策略更新与价值函数更新。
    4. 通过与 REINFORCE 的对比实验，理解 Critic 在方差缩减上的价值。

## 1. 从策略梯度到 Actor-Critic

!!! info "为什么需要 Actor-Critic？"
    在上一任务多臂老虎机中，问题没有"状态"。但真实强化学习场景中，智能体的决策依赖当前状态，并且当前行为会影响未来状态与奖励。本节我们从最基础的策略梯度方法出发，逐步引入 Critic（价值函数）来稳定训练。

### 1.1 MDP 与策略梯度回顾

一个马尔可夫决策过程（Markov Decision Process, MDP）由四元组 $(S, A, P, r)$ 描述：状态空间 $S$、动作空间 $A$、转移概率 $P(s'|s,a)$、奖励函数 $r(s,a)$。智能体的目标是学习一个策略 $\pi_\theta(a|s)$，使长期累积折扣奖励 $J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \gamma^t r_t\right]$ 最大化。

策略梯度定理告诉我们：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \Psi_t\right]
$$

其中 $\Psi_t$ 是某种衡量"在状态 $s_t$ 下采取动作 $a_t$ 有多好"的标量。不同的 $\Psi_t$ 选择对应不同的算法。

### 1.2 高方差问题与基线 / 优势函数

最朴素的选择是用蒙特卡洛回报 $G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}$ 作为 $\Psi_t$（这就是 REINFORCE 算法）。它无偏，但**方差很大**——一条轨迹中早期的好/坏奖励会被全部加进每一步的权重里。

一个经典的方差缩减技巧是引入基线 $b(s)$：

$$
\Psi_t = G_t - b(s_t)
$$

可以证明：只要 $b$ 只依赖状态而不依赖动作，扣除基线**不改变梯度的期望**，但能显著降低方差。最自然的基线选择是状态价值函数 $V^\pi(s) = \mathbb{E}_{\pi}[G_t | s_t = s]$，此时

$$
\Psi_t \approx Q^\pi(s_t, a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)
$$

即**优势函数**：动作 $a_t$ 相对于平均水平好多少。

### 1.3 Actor-Critic 的核心思想

问题是：我们不知道真实的 $V^\pi(s)$。Actor-Critic 的思路是：**再训练一个神经网络 $V_w(s)$ 来近似它**。同时，用时序差分（Temporal Difference, TD）误差近似优势：

$$
\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t) \approx A^\pi(s_t, a_t)
$$

由此得到两个损失：

- **Actor 损失**（更新策略 $\pi_\theta$）：
  $$L_{actor} = -\mathbb{E}\left[\log \pi_\theta(a_t|s_t) \cdot \delta_t \right]$$
  其中 $\delta_t$ 视为常数（仅用作权重，不参与 actor 的反向传播）。

- **Critic 损失**（更新价值 $V_w$）：
  $$L_{critic} = \mathbb{E}\left[\left(V_w(s_t) - (r_t + \gamma V_w(s_{t+1}))\right)^2\right]$$
  即让 $V_w$ 逼近 TD 目标。

这就是最简单的 Actor-Critic 算法。Actor 与 Critic 各有独立的优化器，每一步交互之后两个网络同时更新。

!!! question "思考题 1"
    在策略梯度方法中，直接使用蒙特卡洛回报 $G_t$ 作为权重会带来训练不稳定的问题。请结合"方差"的角度分析：为什么引入一个 Critic 网络（学习 $V(s)$ 或 $A(s,a)$）能够缓解这个问题？

## 2. 环境准备

!!! info "CartPole-v0 简介"
    CartPole-v0 是 OpenAI Gym 提供的经典控制任务：智能体需要左右推动一辆小车，让车顶的一根杆保持竖直。

    - **状态空间**：4 维连续向量（小车位置、小车速度、杆的角度、杆的角速度）
    - **动作空间**：2 个离散动作（向左推 / 向右推）
    - **奖励**：每一步杆未倒下即 +1
    - **终止条件**：杆倾倒超过阈值、小车越出边界，或步数达到 200

!!! warning "依赖安装"
    本任务依赖 `gym`，且推荐使用 0.21.0 版本（与教程接口一致）：
    ```bash
    pip install gym==0.21.0
    ```

首先导入所需模块：

```python
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
```

其中 `rl_utils` 是本实验目录下的工具包，提供 `train_on_policy_agent` 和 `moving_average` 两个函数。

## 3. 策略网络 PolicyNet

策略网络 Actor 是一个简单的两层 MLP，输入是状态，输出是**动作的概率分布**。

请你按照要求补全代码：

```python
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # TODO 1: 输出动作的概率分布
        # 要求: 在动作维度（dim=1）使用 softmax 将 fc2 的输出转为概率
        return ...
```

!!! question "思考题 2"
    Actor 输出层使用 softmax 而不是直接输出动作的值（如 Q 网络那样）。请说明：

    (a) 为什么策略网络的输出需要是一个概率分布？

    (b) 这与基于价值的方法（如 DQN 选 argmax）在"探索"上有什么本质区别？

## 4. 价值网络 ValueNet

价值网络 Critic 同样是一个两层 MLP，但输出是**单个标量** $V(s)$。它的结构非常简单，直接给出：

```python
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

## 5. Actor-Critic 主类

`ActorCritic` 类把 Actor 与 Critic 包装在一起，提供两个核心方法：

- `take_action(state)`：根据当前策略采样一个动作（用于与环境交互）；
- `update(transition_dict)`：用一条完整轨迹的数据同时更新 Actor 与 Critic。

请按要求补全代码：

```python
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, gamma, device):
        # 策略网络（Actor）
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络（Critic）
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # 两个独立的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                 lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                  lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # TODO 2: 根据策略输出的概率分布采样一个动作
        # 提示: 使用 torch.distributions.Categorical 构造分布并调用 .sample()
        action_dist = ...
        action = ...
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # TODO 3: 计算时序差分目标和时序差分误差
        # 要求:
        #   td_target = r + γ * V(s') * (1 - done)
        #   td_delta  = td_target - V(s)
        td_target = ...
        td_delta = ...

        log_probs = torch.log(self.actor(states).gather(1, actions))

        # TODO 4: 计算 actor_loss 和 critic_loss
        # 要求:
        #   actor_loss  = mean( -log π(a|s) * td_delta )   注意 td_delta 需 detach
        #   critic_loss = mean( MSE(V(s), td_target) )     注意 td_target 需 detach
        actor_loss = ...
        critic_loss = ...

        # 反向传播 + 更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
```

!!! question "思考题 3"
    在 actor_loss 的计算中我们写 `td_delta.detach()`，在 critic_loss 中写 `td_target.detach()`。请回答：

    (a) 如果不加 `.detach()`，分别会发生什么？

    (b) 为什么 Actor 的梯度不应该流回 Critic（反之亦然）？

## 6. 训练 Actor-Critic

设置超参数，创建环境与智能体，然后调用 `rl_utils.train_on_policy_agent` 进行训练：

```python
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = ActorCritic(state_dim, hidden_dim, action_dim,
                    actor_lr, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
```

!!! note "实验要求"
    在 CPU 上训练 1000 episodes 通常需要 1-2 分钟。如果你的环境中 `env.seed(0)` 报错，说明你的 `gym` 版本过高，请参考第 2 节安装 0.21.0 版本。

训练结束后，可视化训练曲线：

```python
episodes_list = list(range(len(return_list)))

# 原始 return 曲线
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

# 移动平均平滑曲线
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()
```

你应该会看到：约 400 episodes 之后，return 稳定接近 200（CartPole-v0 的最高分）。

## 7. 对比实验：REINFORCE vs Actor-Critic

为了直观地理解 Critic 带来的方差缩减效果，我们实现一个**简化版 REINFORCE**——它只有 Actor，没有 Critic，使用整轨迹的蒙特卡洛回报 $G_t$ 作为权重，然后与 Actor-Critic 在相同的随机种子、相同的超参数下做对比。

### 7.1 实现 SimplifiedREINFORCE

```python
class SimplifiedREINFORCE:
    """简化版 REINFORCE：只有 Actor，无 Critic，使用蒙特卡洛回报作为权重"""
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item()

    def update(self, transition_dict):
        rewards = transition_dict['rewards']
        states  = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)

        # TODO 5: 反向累加计算每一步的折扣回报 G_t
        # 要求: 从最后一步往前累加，G_t = r_t + γ * G_{t+1}
        # 提示: 倒序遍历 rewards
        G = 0
        returns = []
        for r in rewards[::-1]:
            G = ...
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float).view(-1, 1).to(self.device)

        log_probs = torch.log(self.actor(states).gather(1, actions))
        # 用 returns 替代 td_delta 作为权重
        loss = torch.mean(-log_probs * returns)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
```

### 7.2 训练 REINFORCE

```python
# 重置随机种子，确保与 Actor-Critic 的训练条件一致
torch.manual_seed(0)
env.seed(0)

reinforce_agent = SimplifiedREINFORCE(state_dim, hidden_dim, action_dim,
                                       actor_lr, gamma, device)
reinforce_return_list = rl_utils.train_on_policy_agent(env, reinforce_agent, num_episodes)
```

### 7.3 双曲线对比可视化

```python
episodes_list = list(range(num_episodes))

# 原始 return 曲线对比
plt.plot(episodes_list, return_list, label='Actor-Critic', alpha=0.6)
plt.plot(episodes_list, reinforce_return_list, label='REINFORCE', alpha=0.6)
plt.xlabel('Episodes'); plt.ylabel('Returns'); plt.legend()
plt.title('Actor-Critic vs REINFORCE on CartPole-v0')
plt.show()

# 移动平均平滑后对比
mv_ac = rl_utils.moving_average(return_list, 9)
mv_re = rl_utils.moving_average(reinforce_return_list, 9)
plt.plot(episodes_list, mv_ac, label='Actor-Critic (smoothed)')
plt.plot(episodes_list, mv_re, label='REINFORCE (smoothed)')
plt.xlabel('Episodes'); plt.ylabel('Returns'); plt.legend()
plt.title('Smoothed Comparison')
plt.show()
```

!!! question "思考题 4"
    观察 REINFORCE 与 Actor-Critic 的训练曲线（return 曲线 + 移动平均曲线）。请回答：

    (a) 哪条曲线波动更大？为什么？

    (b) 哪个算法在前 200 episodes 的学习速度更快？请从"每一步是否可以更新"角度解释。

## 8. 总结与反思

本任务中你完成了：

- 从策略梯度出发，理解了 Critic 作为基线降低方差的动机；
- 实现了一个最简单的 Actor-Critic 算法（PolicyNet + ValueNet + TD 更新）；
- 通过与 REINFORCE 的对比实验，直观地看到了 Critic 在训练稳定性和样本效率上的价值。

Actor-Critic 是一个广义的框架——后续许多更先进的算法（A2C / A3C / PPO / SAC 等）都属于这一家族。下一任务 PPO 将在 Actor-Critic 的基础上引入"信任域 / 裁剪"思想，进一步提升训练稳定性。

!!! question "思考题 5"
    Actor-Critic 在 `rl_utils` 中调用的是 `train_on_policy_agent`，每次更新使用的都是"刚刚交互产生的轨迹"。请回答：

    (a) 什么是 on-policy？Actor-Critic 为什么属于 on-policy 算法？

    (b) 在 Actor-Critic 训练中，如果某次更新让策略产生剧烈变化（如学习率过大），可能导致后续采集到的轨迹质量急剧恶化甚至训练崩溃——观察你的训练曲线，是否出现过这种"先升后掉"的现象？请思考：如果想"限制每一步策略更新的幅度"来缓解这一问题，可能需要怎样的设计？（这正是下一任务 PPO 要解决的核心问题）
