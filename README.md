This repository contains an **AlphaZero-inspired chess engine**, developed as part of my MSc thesis.
The project focuses on building a complete and understandable reinforcement learning system, combining a neural network (encoder + policy + value) with Monte Carlo Tree Search (MCTS), trained primarily through self-play under limited computational resources.

**Key Features**

- Neural network with shared encoder + policy and value heads
- Monte Carlo Tree Search guided by neural priors
- Self-play reinforcement learning pipeline
- Supervised warm-start to stabilize early training
- Custom PySide6 GUI for human–AI and AI–AI games
- Clean separation between model, search, training, and UI layers

**Architecture Overview**

Encoder – transforms chess positions into a learned representation
Policy head – predicts a probability distribution over legal moves
Value head – estimates the expected game outcome
MCTS – refines network predictions through guided search

The system **does not** rely on handcrafted chess heuristics and learns entirely from data.

**Training Insights**

Training revealed that AlphaZero-style systems are highly sensitive to instability:
- Naïve scaling of simulations can lead to policy collapse
- Training quality depends more on data generation than raw compute
- Stable learning requires active monitoring and intervention

This effectively turns the process into **Human-in-the-Loop Reinforcement Learning**.

**Results**

Approximate playing strength: ~800 ELO
Stable, non-random gameplay
Fully functional end-to-end system (training → evaluation → play)


**Takeaway**

AlphaZero-like systems do not train on autopilot.
Successful learning is an engineering problem, not just an algorithmic one.
