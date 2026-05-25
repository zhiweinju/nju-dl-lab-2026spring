# 深度学习实验:扩散模型 (Denoising Diffusion Probabilistic Models)

> 实验周期:两周(课余时间)  
> 算力要求:**仅需 CPU**(笔记本电脑可完成,无需 GPU)  
> 提交形式:一份 `.ipynb` 文件

---

## 一、实验目的

1. **理解** DDPM 前向加噪过程(forward diffusion)和反向去噪过程(reverse process)的数学原理;
2. **实现** DDPM 的核心模块,包括:噪声调度、前向采样 $q(x_t \mid x_0)$、训练损失、反向采样 $p_\theta(x_{t-1} \mid x_t)$;
3. **训练** 一个小型扩散模型,在 2D 玩具分布与 MNIST 手写数字数据集上完成生成任务;
4. **分析** 噪声步数 $T$、噪声调度策略、采样方法(DDPM vs. DDIM)对生成质量的影响。


---

## 二、实验背景与核心数学

### 2.1 前向扩散过程

给定一张干净图像 $x_0 \sim q(x_0)$,前向过程在 $T$ 步内不断加入高斯噪声:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t \mathbf{I})
$$

其中 $\{\beta_t\}_{t=1}^{T}$ 是预先设定的噪声调度(noise schedule),常用线性调度 $\beta_t \in [10^{-4}, 0.02]$。

令 $\alpha_t = 1-\beta_t$,$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$,可以直接采样 $x_t$:

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1-\bar{\alpha}_t)\mathbf{I}\big)
$$

即 $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\ \epsilon \sim \mathcal{N}(0,\mathbf{I})$。

### 2.2 训练目标

DDPM 采用一个参数化网络 $\epsilon_\theta(x_t, t)$ 预测加入的噪声 $\epsilon$,损失函数简化为:

$$
\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t,\ x_0,\ \epsilon}\Big[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\ t\big)\big\|^2\Big]
$$

### 2.3 反向采样过程

训练完成后,从 $x_T \sim \mathcal{N}(0,\mathbf{I})$ 出发,按下式逐步去噪:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon_\theta(x_t,t)\right) + \sigma_t z,\quad z\sim \mathcal{N}(0,\mathbf{I})
$$

其中 $\sigma_t^2 = \beta_t$(或 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$),$t=1$ 时 $z=0$。

> 📌 详细推导请参考课程讲义或 Ho et al. 2020《Denoising Diffusion Probabilistic Models》。

---

## 三、实验环境

- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0(CPU 版即可,无需 CUDA)
- **依赖**:`numpy`, `matplotlib`, `tqdm`, `scikit-learn`, `torchvision`
- 建议在 `Jupyter Notebook` 或 `JupyterLab` 中完成

安装命令:

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn jupyter
```

---

## 四、实验内容

实验分为 **4 个任务**,前 3 个为必做,Task 4 为加分项。所有任务在同一份 Notebook(`diffusion_lab_template.ipynb`)中完成。

### Task 1:前向扩散过程的实现与可视化(20 分)

- 实现噪声调度函数(linear schedule)、`alpha_t` 与 `alpha_bar_t` 的计算;
- 实现 `q_sample(x_0, t, noise)`,即从 $q(x_t \mid x_0)$ 采样;
- 取一张 MNIST 图像,可视化 $t \in \{0, 50, 100, 200, 500, 999\}$ 时的 $x_t$;
- **思考题**:为什么大 $t$ 时 $x_t$ 看起来近似纯噪声?用 $\bar{\alpha}_t$ 解释。

### Task 2:2D 玩具数据集上的 DDPM(30 分)

数据:`sklearn` 提供的 Swiss Roll(三维曲面投影到 2D)与 Two Moons。

要求:
- 实现 `train_loss(model, x0)`,计算 $\mathcal{L}_{\text{simple}}$;
- 实现 `ddpm_sample(model, n_samples)`,按 §2.3 公式从噪声采样;
- 用模板提供的小型 MLP(约 1 万参数)训练,**预计 CPU 上 1–3 分钟**;
- 绘制训练数据 vs. 生成样本的散点图,验证模型确实学到了目标分布;
- **思考题**:把 $T$ 从 1000 改为 50,生成质量如何变化?为什么?

### Task 3:MNIST 上的 DDPM(40 分)

数据:`torchvision.datasets.MNIST`(自动下载),为节省 CPU 时间,实验中将图像 **下采样到 16×16**,并仅使用 **训练集前 10000 张**。

要求:
- 使用模板提供的小型 UNet(约 30 万参数,已实现);
- 实现训练循环,**预计 CPU 上 30–60 分钟,15 epoch 内可见效果**;
- 用训练好的模型采样 64 张 16×16 图像,以 8×8 网格展示;
- 给出训练 loss 曲线;
- **思考题**:报告中至少包含一组生成图像,并讨论失败/成功的样本。

### Task 4(加分,10 分):DDIM 加速采样

- 实现 DDIM 的确定性采样(Song et al. 2021),仅使用 $S = 20$ 步采样;
- 比较 DDPM(1000 步) 与 DDIM($S=20$) 的生成质量和耗时;
- **思考题**:为什么 DDIM 可以跳步而 DDPM 不行?

---

## 五、数据集

### 5.1 MNIST(主数据集)

代码模板中已通过 `torchvision.datasets.MNIST(..., download=True)` 自动下载。

如自动下载失败,可手动从以下任一镜像获取:

| 来源 | 链接 |
| --- | --- |
| Yann LeCun 原始站点 | `https://web.archive.org/web/2024/http://yann.lecun.com/exdb/mnist/` |
| HuggingFace Datasets | `https://huggingface.co/datasets/ylecun/mnist` |
| kaggle | `https://www.kaggle.com/datasets/hojjatk/mnist-dataset` |

下载得到 4 个 `.gz` 文件,放入 `./data/MNIST/raw/` 目录即可。

### 5.2 Swiss Roll / Two Moons(Task 2 玩具数据)

由 `sklearn.datasets.make_swiss_roll` 与 `make_moons` 在线生成,**无需下载**。

---

## 六、代码模板下载

[下载代码模板文件](https://cdn.jsdelivr.net/gh/zhiweinju/nju-dl-lab-2026spring@main/docs/lab11/diffusion_lab_template.ipynb){ .md-button}

模板中:

- **已给出**:数据加载、UNet 架构、训练循环骨架、可视化绘图函数;

- **需补全**:噪声调度、`q_sample`、训练损失、`ddpm_sample`、(可选)`ddim_sample`,以及若干思考题的文字回答。

代码中所有需要补全的位置已用 `# TODO:` 注释标出。

---

## 七、提交要求

1. 提交单个 `.ipynb` 文件,命名为 `学号_姓名_diffusion_lab.ipynb`;
2. 提交前请 **重新运行所有 cell**(`Kernel → Restart & Run All`),确保:
    - 所有可视化结果可见;
    - 训练 loss 曲线已绘出;
    - 生成图像已展示;
3. 思考题的文字回答写在实验报告中,字数不强制要求,**但需说清原因**;
4. **不要** 提交 `data/` 目录或模型权重文件。

---

## 八、评分标准

| 项目 | 分值 |
| --- | --- |
| Task 1:前向过程实现 + 可视化 | 20 |
| Task 2:2D 数据 DDPM 完整闭环 | 30 |
| Task 3:MNIST DDPM 训练 + 生成 | 40 |
| 思考题回答质量 | 10 |
| Task 4 (加分):DDIM 加速采样 | +10 |
| 代码可读性与注释 | 含在以上各项中 |

⚠️ **学术诚信**:本实验允许查阅论文与公开教程,但提交代码必须由本人独立编写;严禁直接复制他人的实验代码或网络上的 DDPM 教程代码。

---

## 九、参考资料

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. arXiv:2006.11239
2. Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR. arXiv:2010.02502
3. Stanford CS231n (2026 Spring) Assignment 3 — DDPM 部分: <https://cs231n.github.io/assignments2026/assignment3/>
4. UC Berkeley CS294-158 Deep Unsupervised Learning(Diffusion Models 章节): <https://sites.google.com/view/berkeley-cs294-158-sp24/home>
5. Lilian Weng 博文:*What are Diffusion Models?* <https://lilianweng.github.io/posts/2021-07-11-diffusion-models/>

---

祝实验顺利!有疑问请在课程答疑群或与助教联系。
