# 实验任务三：TRPO 与 PPO算法

本书之前介绍的基于策略的方法包括策略梯度算法和 Actor-Critic 算法。这些方法虽然简单、直观，但在实际应用过程中会遇到训练不稳定的情况。回顾一下基于策略的方法：参数化智能体的策略，并设计衡量策略好坏的目标函数，通过梯度上升的方法来最大化这个目标函数，使得策略最优。具体来说，假设 $\theta$ 表示策略 $\pi_\theta$ 的参数，定义

$$
J(\theta) = \mathbb{E}_{s_0} [V^{\pi_\theta}(s_0)] = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]
$$

基于策略的方法的目标是找到 $\theta^* = \arg \max_{\theta} J(\theta)$，策略梯度法主要沿着 $\nabla_\theta J(\theta)$ 方向迭代更新策略参数 $\theta$。但是这种算法有一个明显的缺点：当策略网络是深度模型时，沿着策略梯度更新参数，很有可能由于步长太长，策略突然显著变差，进而影响训练效果。

针对以上问题，我们考虑在更新时找到一块信任区域（trust region），在这个区域上更新策略时能够得到某种策略性能的安全性保证，这就是信任区域策略优化（trust region policy optimization，TRPO）算法的主要思想。TRPO 算法在 2015 年被提出，它在理论上能够保证策略学习的性能单调性，并在实际应用中取得了比策略梯度算法更好的效果。


TRPO（Trust Region Policy Optimization，信任区域策略优化）和 PPO（Proximal Policy Optimization，近端策略优化）是深度强化学习中非常重要的策略梯度类算法。它们旨在解决传统策略梯度方法（如 REINFORCE）中**步长难以选择**、**训练不稳定**的问题，通过限制策略更新的幅度来提升学习的鲁棒性。

<span style="color:red;"> TRPO 和PPO算法是比较难掌握的一种强化学习算法，需要较好的数学基础。例如1.3节的求解方法我们在此不在具体展开，同学若在学习过程中遇到困难，可自行查阅相关资料。在这一章节中, 同学们首先学习TRPO算法的基本原理，然后完成一个简单的思考题，并PPO-clip算法的take_action函数部分。</span> 

---

## 1. TRPO 算法

### 1.1 核心思想

TRPO 的核心是**在每次迭代中，找到一个新策略，使得目标函数单调不减，同时保证新旧策略之间的差异不超过一个预设的“信任区域”**。它通过 KL 散度约束策略更新的大小，避免因单步更新过大导致性能崩溃。

### 1.2 数学形式

TRPO 解决如下约束优化问题：


\begin{aligned}
\max_{\theta} \quad & \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}, a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) \right] \\
\text{s.t.} \quad & \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}} \left[ D_{KL}\left( \pi_{\theta_{\text{old}}}(\cdot|s) \;\|\; \pi_{\theta}(\cdot|s) \right) \right] \leq \delta
\end{aligned}


其中：
- $\pi_{\theta}$ 为策略网络。
- $A^{\pi_{\theta_{\text{old}}}}(s,a)$ 是优势函数。
- $\rho_{\pi_{\theta_{\text{old}}}}$ 是旧策略下的状态分布。
- $D_{KL}$ 是 KL 散度，$\delta$ 为信任区域半径（超参数）。

### 1.3 求解方法

直接求解带约束的优化问题较困难。TRPO 通常采用：
1. 对目标函数做**一阶近似**。
2. 对约束条件做**二阶近似**（使用 Fisher 信息矩阵）。
3. 通过**共轭梯度法**近似计算自然梯度方向。
4. 进行**线性搜索**，确保满足 KL 约束并提升目标函数。

### 1.4 代码实践

本节将使用支持与离散和连续两种动作交互的环境来进行 TRPO 的实验。我们使用的第一个环境是车杆（CartPole），第二个环境是倒立摆（Inverted Pendulum）。这两个环境的细节如下：


<span style="color:blue;"> CartPole（车杆 / 小车倒立摆）</span>

**环境描述**：一个小车可以在水平轨道上左右移动，一个杆子通过转轴固定在小车上。杆子初始时通常是竖直向下（或略微倾斜），目标是通过给小车施加左右方向的力，使杆子保持直立（不倒）。

**状态空间**：通常为 4 维连续值：

- 小车位置 $x$
- 小车速度 $\dot{x}$
- 杆子角度 $\theta$（竖直向上为 0）
- 杆子角速度 $\dot{\theta}$

**动作空间**：离散，2 个动作（向左用力 / 向右用力）。

**终止条件**：杆子倾斜角度过大（超过 ±12°）、小车移出轨道边缘，或达到最大步数。

**典型环境**：OpenAI Gym 中的 `CartPole-v1`。

<span style="color:blue;"> 倒立摆（Inverted Pendulum）</span>

**环境描述**：一个摆杆的一端固定在中心，另一端自由。目标是通过对中心点施加扭矩，使摆杆从任意初始角度（通常是下垂或随机）摆起并稳定在竖直向上位置（$\theta = \pi$）。

**状态空间**：3 维连续值：

- $\cos\theta$（摆角余弦）
- $\sin\theta$（摆角正弦）
- 角速度 $\dot{\theta}$

**动作空间**：连续值，范围 $[-2, 2]$，表示施加的扭矩。

**奖励函数**：通常为 $-\left(\theta^2 + 0.1 \dot{\theta}^2 + 0.001 \cdot \text{action}^2\right)$，鼓励角度接近竖直、角速度小、动作小。

**终止条件**：无固定终止步数，通常训练到一定步数（如 200 步）结束。

在了解完环境后，我们正式开始代码实践。首先导入一些必要的库。

```python
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import rl_utils
import copy
```
然后定义策略网络和价值网络（与 Actor-Critic 算法一样）


```python
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

进而我们需要定义TRPO算法，这里我们给出了完整的代码，同学们可以学习policy_learn函数的作用。

```python
class TRPO:
    """ TRPO算法 """
    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.n
        # 策略网络参数不需要优化器更新
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):  
        # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):  
        # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
         # 线性搜索主循环
        for i in range(15): 
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = 
            kl_div =
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage):  # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)  # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略

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
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                      td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)
```

接下来我们将在车杆环境中训练 TRPO，并将结果可视化。


```python
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
critic_lr = 1e-2
kl_constraint = 0.0005
alpha = 0.5
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
             kl_constraint, alpha, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('TRPO on {}'.format(env_name))
plt.show()

```

TRPO 在车杆环境中很快收敛，展现了十分优秀的性能效果。 

思考题1: 在倒立摆环境中，我们需要对策略函数做哪些修改？

## 2. PPO 算法

PPO 由 OpenAI 于 2017 年提出，旨在保留 TRPO 稳定性的同时，大幅简化实现。它放弃了严格的 KL 约束，改用**裁剪（clip）**或**自适应 KL 惩罚**的方式间接限制更新幅度。

### 2.1 核心思想

PPO 的关键是定义一个**截断的替代目标函数**，当新旧策略的比值 $r(\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 偏离 1 太远时，自动截断该样本的贡献，从而避免过大更新。

### 2.2 PPO-Clip（最常用版本）

目标函数为：


$L^{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$


其中：
- $r_t(\theta)$是新旧策略概率比。
- $A_t$ 是时刻 $t$ 的优势函数估计。
- $\epsilon$ 是超参数（通常取 0.1 或 0.2）。
- $\text{clip}(x, 1-\epsilon, 1+\epsilon)$ 将 $x$ 限制在 $[1-\epsilon, 1+\epsilon]$ 内。

**直观理解**：
- 当 $A_t > 0$（动作好），我们希望增大该动作的概率，但若 $r(\theta) > 1+\epsilon$，裁剪到 $1+\epsilon$，防止步子迈太大。
- 当 $A_t < 0$（动作差），我们希望减小概率，但若 $r(\theta) < 1-\epsilon$，裁剪到 $1-\epsilon$，避免过度惩罚。

### 2.3 PPO 的其他变体

- **PPO-KL Penalty**：在目标函数中增加一个自适应系数的 KL 惩罚项，而非硬约束。
- 实际中 **PPO-Clip** 更简单有效，成为主流。

---

### 2.4 代码实践
与 TRPO 相同，我们仍然可以在车杆和倒立摆两个环境中测试PPO-Clip算法。

与TRPO首先导入一些必要的库，并定义策略网络和价值网络。

```python
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```


接下来请同学们完成PPO-clip算法的take_action函数部分。

```python
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):


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
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
```

接下来在车杆环境中训练 PPO 算法。

```python
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
```

最后我们进行一个可视化

```python
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()
```

## 3. 总结

- **TRPO** 是理论奠基之作，引入了信任区域的思想，通过严格约束 KL 散度保证策略单调改进，但实现复杂、计算量大。
- **PPO** 是 TRPO 的工程化简化版本，通过精巧的裁剪机制达到了与 TRPO 相近甚至更好的稳定性，同时代码简洁、易于扩展，已成为目前强化学习中最主流的策略优化算法之一。