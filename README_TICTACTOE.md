# 井字棋AI模型训练项目

这个项目实现了一个基于Q-learning强化学习算法的井字棋(Tic-Tac-Toe)AI模型训练系统。

## 功能特点

- 🎮 完整的井字棋游戏逻辑实现
- 🤖 基于Q-learning的强化学习AI智能体
- 📊 训练过程统计和可视化
- 💾 模型保存和加载功能
- 🎯 人机对战模式
- 📈 性能评估系统

## 项目结构

```
.
├── tic_tac_toe/          # 核心代码包
│   ├── __init__.py       # 包初始化
│   ├── game.py           # 游戏逻辑
│   ├── agent.py          # AI智能体
│   └── trainer.py        # 训练器
├── train_model.py        # 基础模型训练脚本
├── train_advanced.py     # 高级训练脚本 (自我对弈，提升胜率)
├── train_ultra.py        # 超级训练脚本 (集成所有优化方法) 🚀
├── play_game.py          # 人机对战脚本
└── README_TICTACTOE.md   # 项目文档
```

## 安装依赖

```bash
pip install numpy
```

## 使用方法

### 1. 训练模型

#### 基础训练 (对抗随机对手)

使用默认参数训练模型：

```bash
python train_model.py
```

自定义训练参数：

```bash
python train_model.py --episodes 50000 --learning-rate 0.1 --epsilon 0.15
```

可用参数：
- `--episodes`: 训练轮数 (默认: 20000)
- `--learning-rate`: 学习率 (默认: 0.1)
- `--discount-factor`: 折扣因子 (默认: 0.9)
- `--epsilon`: 探索率 (默认: 0.1)
- `--output`: 模型保存路径 (默认: models/tic_tac_toe_model.pkl)
- `--eval-games`: 评估游戏数 (默认: 100)

**预期效果:** 胜率约 45-48%

#### 高级训练 (自我对弈，提升胜率) 🌟

使用自我对弈和渐进式训练策略：

```bash
python train_advanced.py
```

自定义训练参数：

```bash
python train_advanced.py --episodes 30000 --learning-rate 0.15
```

可用参数：
- `--episodes`: 总训练轮数 (默认: 30000)
- `--learning-rate`: 学习率 (默认: 0.15)
- `--discount-factor`: 折扣因子 (默认: 0.95)
- `--output`: 模型保存路径 (默认: models/advanced_model.pkl)
- `--eval-games`: 评估游戏数 (默认: 200)

**训练策略:**
1. 阶段1: 高探索率 (ε=0.3) - 两个AI互相对弈探索策略
2. 阶段2: 中等探索率 (ε=0.15) - 平衡探索和利用
3. 阶段3: 低探索率 (ε=0.05) - 精细调优

**预期效果:** 胜率约 50-60%+

#### 超级训练 (集成所有优化方法) 🚀

使用状态对称性、双Q学习、经验回放和集成学习：

```bash
python train_ultra.py
```

自定义训练参数：

```bash
python train_ultra.py --episodes 30000 --num-models 3
```

可用参数：
- `--episodes`: 每个模型的训练轮数 (默认: 30000)
- `--num-models`: 集成模型数量 (默认: 3)
- `--output-dir`: 模型保存目录 (默认: models/ultra)

**优化方法:**
1. **状态对称性利用**: 8种对称状态共享Q值，效率提升8倍
2. **双Q学习**: 使用两个Q表减少过高估计
3. **经验回放**: 存储历史对局，随机采样重新学习
4. **集成学习**: 训练多个模型，投票决策

**预期效果:** 胜率约 70-80%+

### 2. 与AI对战

#### 可视化人机对战 (推荐) 🎮

使用美化的彩色界面与AI对战：

```bash
# 使用最佳模型 (72.5%胜率)
python play_visual.py

# 自定义模型
python play_visual.py --model models/my_model.pkl

# AI先手
python play_visual.py --ai-first
```

**界面特点:**
- 🎨 彩色棋盘显示 (X=红色, O=蓝色)
- 📍 清晰的坐标提示
- 💡 友好的用户提示和帮助
- 📊 实时游戏统计
- 🔄 支持多局连续对战

#### 简单命令行对战

使用简单文本界面：

```bash
python play_game.py
```

让AI先手：

```bash
python play_game.py --ai-first
```

### 3. 对局分析与复盘 📊

分析AI输掉的对局，找出弱点并提供改进建议：

```bash
# 分析模型性能
python analyze_games.py --model models/ultra_trained/model_1.pkl

# 分析更多对局
python analyze_games.py --model models/my_model.pkl --games 200

# 指定输出文件
python analyze_games.py --output my_analysis.txt
```

**分析功能:**
- 📊 统计胜率、失败率、平局率
- 🔍 分析失败模式和原因
- ⚠️  识别关键失误（如错过获胜机会）
- 🎯 识别弱势位置和状态
- 💡 提供针对性改进建议
- 💾 生成详细分析报告

**分析报告包含:**
- 失败对局的棋盘状态
- 关键失误的具体走法
- 最常失败的状态分析
- 优先级排序的改进建议

### 4. 游戏规则

- 棋盘是3x3的网格
- 玩家使用 X，AI使用 O（或相反，取决于谁先手）
- 输入格式：`行 列`（例如：`0 0` 表示左上角）
- 行和列的范围都是 0-2
- 先连成三个的一方获胜

棋盘坐标示例：
```
  0 1 2
0 . . .
1 . . .
2 . . .
```

## 算法说明

### Q-Learning算法

本项目使用Q-learning强化学习算法训练AI。Q-learning是一种无模型的强化学习算法，通过学习状态-动作对的价值（Q值）来做出最优决策。

**核心概念：**
- **Q值**: 在某个状态下执行某个动作的预期累积奖励
- **学习率(α)**: 控制新信息对Q值更新的影响程度
- **折扣因子(γ)**: 控制未来奖励的重要性
- **探索率(ε)**: epsilon-greedy策略中的探索概率

**更新公式：**
```
Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

### 训练策略

1. **自我对弈**: AI通过与随机对手对弈来学习
2. **探索与利用**: 使用epsilon-greedy策略平衡探索新策略和利用已知最优策略
3. **奖励设置**:
   - 获胜: +1.0
   - 失败: -1.0
   - 平局: +0.5
   - 中间步骤: 0.0

## 性能优化建议

- 增加训练轮数可以提高模型性能
- 调整学习率和探索率以获得更好的收敛效果
- 可以实现AI自我对弈以进一步提升性能

## 代码示例

### 使用代码训练模型

```python
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer

# 创建智能体
agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.1)
opponent = RandomAgent(player=-1)

# 创建训练器并训练
trainer = TicTacToeTrainer()
trainer.train(agent, opponent, num_episodes=10000)

# 保存模型
agent.save_model('my_model.pkl')

# 评估性能
trainer.evaluate(agent, opponent, num_games=100)
```

### 加载模型进行预测

```python
from tic_tac_toe import QLearningAgent, TicTacToe

# 加载模型
agent = QLearningAgent()
agent.load_model('models/tic_tac_toe_model.pkl')

# 创建游戏
game = TicTacToe()
game.reset()

# 让AI选择动作
action = agent.choose_action(game, training=False)
print(f"AI选择的动作: {action}")
```

## 技术栈

- **Python 3**: 编程语言
- **NumPy**: 数值计算
- **Pickle**: 模型序列化

## 扩展建议

可以在此基础上进行以下扩展：

1. 实现深度Q网络(DQN)版本
2. 添加图形界面
3. 支持更大的棋盘(如5x5的五子棋)
4. 实现其他强化学习算法(如SARSA、Actor-Critic等)
5. 添加对战历史记录和回放功能

## 贡献

欢迎提交Issue和Pull Request！

## 许可

MIT License
