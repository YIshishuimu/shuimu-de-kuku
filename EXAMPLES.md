# 井字棋AI使用示例 / Tic-Tac-Toe AI Usage Examples

## 目录
- [快速开始](#快速开始)
- [基本使用](#基本使用)
- [高级用法](#高级用法)
- [API参考](#api参考)

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行演示
```bash
python demo.py
```

### 3. 训练模型

#### 基础训练
```bash
python train_model.py
```

#### 高级训练 (提升胜率) 🌟
```bash
python train_advanced.py
```

### 4. 与AI对战
```bash
python play_game.py
```

## 基本使用

### 训练一个简单的模型

```python
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer

# 创建Q-learning智能体
agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.1)

# 创建随机对手
opponent = RandomAgent(player=-1)

# 创建训练器
trainer = TicTacToeTrainer()

# 训练10000轮
trainer.train(agent, opponent, num_episodes=10000)

# 保存模型
agent.save_model('my_model.pkl')
```

### 加载并使用训练好的模型

```python
from tic_tac_toe import QLearningAgent, TicTacToe

# 加载模型
agent = QLearningAgent()
agent.load_model('my_model.pkl')

# 创建游戏
game = TicTacToe()
game.reset()

# 让AI选择最佳动作
action = agent.choose_action(game, training=False)
print(f"AI选择: {action}")

# 执行动作
success, winner, done = game.make_move(action)
```

### 评估模型性能

```python
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer

# 加载模型
agent = QLearningAgent()
agent.load_model('my_model.pkl')

# 创建对手
opponent = RandomAgent(player=-1)

# 评估
trainer = TicTacToeTrainer()
agent.set_epsilon(0.0)  # 关闭探索
trainer.evaluate(agent, opponent, num_games=100)
```

## 高级用法

### 使用高级训练脚本 (自我对弈提升胜率) 🌟

最简单的方法是使用内置的高级训练脚本：

```bash
# 使用默认参数 (30000轮训练)
python train_advanced.py

# 自定义参数
python train_advanced.py --episodes 50000 --learning-rate 0.15
```

**工作原理:**
- 两个Q-learning智能体互相对弈
- 渐进式降低探索率 (0.3 → 0.15 → 0.05)
- 自动选择表现更好的智能体保存

**预期结果:** 胜率从 45% 提升到 50-60%+

### 自定义训练参数

```bash
python train_model.py \
    --episodes 50000 \
    --learning-rate 0.15 \
    --discount-factor 0.95 \
    --epsilon 0.2 \
    --output models/advanced_model.pkl
```

### 两个Q-learning智能体互相训练 (代码示例)

```python
from tic_tac_toe import QLearningAgent, TicTacToeTrainer

# 创建两个智能体
agent1 = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.1)
agent2 = QLearningAgent(player=-1, learning_rate=0.1, epsilon=0.1)

# 训练
trainer = TicTacToeTrainer()
trainer.train(agent1, agent2, num_episodes=20000)

# 两个智能体都会学习
print(f"Agent1 Q-table size: {len(agent1.q_table)}")
print(f"Agent2 Q-table size: {len(agent2.q_table)}")
```

### 渐进式训练策略

```python
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer

agent = QLearningAgent(player=1, learning_rate=0.1, epsilon=0.3)
opponent = RandomAgent(player=-1)
trainer = TicTacToeTrainer()

# 阶段1: 高探索率
print("阶段1: 探索...")
agent.set_epsilon(0.3)
trainer.train(agent, opponent, num_episodes=5000)

# 阶段2: 中等探索率
print("阶段2: 平衡探索和利用...")
agent.set_epsilon(0.1)
trainer.reset_stats()
trainer.train(agent, opponent, num_episodes=10000)

# 阶段3: 低探索率精细调优
print("阶段3: 利用已学知识...")
agent.set_epsilon(0.01)
trainer.reset_stats()
trainer.train(agent, opponent, num_episodes=5000)

# 保存最终模型
agent.save_model('progressive_model.pkl')
```

### 可视化游戏过程

```python
from tic_tac_toe import TicTacToe, QLearningAgent, RandomAgent

# 加载模型
agent = QLearningAgent()
agent.load_model('my_model.pkl')
agent.set_epsilon(0.0)

opponent = RandomAgent(player=-1)

# 创建游戏并可视化
game = TicTacToe()
game.reset()

print("游戏开始！")
game.render()

move_count = 0
while True:
    current_player = game.current_player
    
    if current_player == 1:
        action = agent.choose_action(game, training=False)
        print(f"AI (X) 选择: {action}")
    else:
        action = opponent.choose_action(game)
        print(f"随机对手 (O) 选择: {action}")
    
    success, winner, done = game.make_move(action)
    move_count += 1
    
    game.render()
    
    if done:
        if winner == 0:
            print("平局！")
        elif winner == 1:
            print("AI (X) 获胜！")
        else:
            print("随机对手 (O) 获胜！")
        break

print(f"总共 {move_count} 步")
```

### 批量训练和比较

```python
from tic_tac_toe import QLearningAgent, RandomAgent, TicTacToeTrainer
import matplotlib.pyplot as plt

# 测试不同的学习率
learning_rates = [0.05, 0.1, 0.15, 0.2]
results = {}

for lr in learning_rates:
    print(f"\n训练学习率 = {lr}")
    agent = QLearningAgent(player=1, learning_rate=lr, epsilon=0.1)
    opponent = RandomAgent(player=-1)
    trainer = TicTacToeTrainer()
    
    trainer.train(agent, opponent, num_episodes=5000, print_interval=5000)
    
    # 评估
    agent.set_epsilon(0.0)
    wins = trainer.evaluate(agent, opponent, num_games=100, verbose=False)
    results[lr] = wins[1]  # 记录胜率

# 打印比较结果
print("\n学习率比较:")
for lr, wins in results.items():
    print(f"  学习率 {lr}: 胜率 {wins}%")
```

## API参考

### TicTacToe类

```python
game = TicTacToe()

# 重置游戏
game.reset()

# 获取可用动作
actions = game.get_available_actions()  # [(0,0), (0,1), ...]

# 执行动作
success, winner, done = game.make_move((0, 0))

# 显示棋盘
game.render()

# 检查获胜者
winner = game.check_winner()  # 1, -1, 0, 或 None
```

### QLearningAgent类

```python
agent = QLearningAgent(
    player=1,              # 1 或 -1
    learning_rate=0.1,     # 学习率 (0-1)
    discount_factor=0.9,   # 折扣因子 (0-1)
    epsilon=0.1            # 探索率 (0-1)
)

# 选择动作
action = agent.choose_action(game, training=True)

# 设置探索率
agent.set_epsilon(0.05)

# 保存/加载模型
agent.save_model('model.pkl')
agent.load_model('model.pkl')
```

### TicTacToeTrainer类

```python
trainer = TicTacToeTrainer()

# 训练
trainer.train(agent1, agent2, num_episodes=10000, print_interval=1000)

# 评估
wins = trainer.evaluate(agent1, agent2, num_games=100, verbose=False)

# 重置统计
trainer.reset_stats()
```

## 命令行工具

### train_model.py

```bash
# 基本用法
python train_model.py

# 自定义参数
python train_model.py --episodes 30000 --learning-rate 0.15

# 完整参数
python train_model.py \
    --episodes 50000 \
    --learning-rate 0.1 \
    --discount-factor 0.9 \
    --epsilon 0.1 \
    --output models/my_model.pkl \
    --eval-games 200
```

### play_game.py

```bash
# 默认模型
python play_game.py

# 指定模型
python play_game.py --model models/my_model.pkl

# AI先手
python play_game.py --ai-first
```

## 故障排除

### 模型性能不佳
- 增加训练轮数 (`--episodes`)
- 调整学习率 (`--learning-rate`)
- 增加探索率 (`--epsilon`)
- 使用渐进式训练策略

### 训练速度慢
- 减少 `print_interval`
- 使用较少的训练轮数进行初步测试
- 考虑使用更简单的对手

### 内存使用过多
- Q表会随着探索的状态增加而增长
- 考虑实现状态聚合或函数逼近方法

## 更多资源

- [Q-learning算法详解](https://en.wikipedia.org/wiki/Q-learning)
- [强化学习入门](https://www.google.com)
- 项目主文档: [README_TICTACTOE.md](README_TICTACTOE.md)
