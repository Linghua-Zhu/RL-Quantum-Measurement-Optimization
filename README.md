# Artificial-Intelligence-Driven Shot Reduction in Quantum Measurement

## Overview
This repository contains the source code related to the paper titled "Artificial-Intelligence-Driven Shot Reduction in Quantum Measurement". This project employs a reinforcement learning strategy to dynamically optimize shot allocation in Variational Quantum Eigensolvers (VQE), reducing the number of quantum measurements needed while maintaining computational accuracy.


![AI-Driven VQE Diagram](https://github.com/Linghua-Zhu/RL-Quantum-Measurement-Optimization/blob/main/images/AIvqe-1.png)

## Paper
For detailed information, you can access the paper here: [(https://arxiv.org/abs/2405.02493)] 

## Features
- **Dynamic Shot Allocation**: Implements a reinforcement learning approach to adjust measurement shots during VQE optimization.
- **Enhanced Efficiency**: Reduces the number of shots required, lowering computational costs.
- **Broad Applicability**: Demonstrates transferability across various molecular systems and compatibility with multiple wavefunction ansatzes.

## Prerequisites
To use this software, ensure you have the following installed:
- Python 3.8+
- Libraries: numpy, scipy, qiskit

## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Linghua-Zhu/RL-Quantum-Measurement-Optimization.git
cd RL-Quantum-Measurement-Optimization
pip install -r requirements.txt
