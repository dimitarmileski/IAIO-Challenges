import os
import json

# 1. Define the Problem Statement Content
problem_text = """======================================================================
CEOAI / EUROAI 2026 Challenge: Subgrid Mine-Risk Assessment (SMRA)
======================================================================

TASK DESCRIPTION:
In automated puzzle solvers, determining the safety index of an unrevealed 
grid tile based on adjacent conditions is a highly non-linear classification 
task. In this problem, you are given partial configurations of 9x9 Minesweeper 
boards. Each board state is provided as a grid where some cells are revealed, 
some cells are explicitly flagged as mines (-2), and others remain hidden (-1).

Your objective is to predict whether a specific targeted hidden tile contains 
a mine (1) or is safe (0).

DATASET STRUCTURE:
- board_state: A flattened 81-element string representing the 9x9 grid row-by-row.
- target_row & target_col: The 0-indexed coordinate of the hidden cell to evaluate.
- is_mine: (Train only) Binary label where 1 represents a mine, and 0 indicates safe.

EVALUATION METRIC:
Submissions will be evaluated using the Macro F1-Score.
"""

# 2. Define the Starter Notebook Structure
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CEOAI / EUROAI Starter Sandbox Environment\n",
                "## Task: Subgrid Mine-Risk Assessment"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from sklearn.metrics import f1_score\n",
                "\n",
                "BATCH_SIZE = 64\n",
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Executing training workflow on device: {DEVICE}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class MinesweeperDataset(Dataset):\n",
                "    def __init__(self, df, is_test=False):\n",
                "        self.df = df.reset_index(drop=True)\n",
                "        self.is_test = is_test\n",
                "        \n",
                "    def __len__(self):\n",
                "        return len(self.df)\n",
                "        \n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        grid = np.array([int(x) for x in row['board_state'].split()]).reshape(9, 9)\n",
                "        \n",
                "        mask = np.zeros((9, 9))\n",
                "        mask[int(row['target_row']), int(row['target_col'])] = 1.0\n",
                "        \n",
                "        features = np.stack([grid, mask], axis=0).astype(np.float32)\n",
                "        \n",
                "        if self.is_test:\n",
                "            return torch.tensor(features)\n",
                "        else:\n",
                "            label = int(row['is_mine'])\n",
                "            return torch.tensor(features), torch.tensor(label, dtype=torch.long)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class BoardEvaluatorCNN(nn.Module):\n",
                "    def __init__(self):\n",
                "        super(BoardEvaluatorCNN, self).__init__()\n",
                "        self.conv_block = nn.Sequential(\n",
                "            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),\n",
                "            nn.BatchNorm2d(32),\n",
                "            nn.ReLU(),\n",
                "            nn.Flatten()\n",
                "        )\n",
                "        self.classifier = nn.Sequential(\n",
                "            nn.Linear(32 * 9 * 9, 128),\n",
                "            nn.ReLU(),\n",
                "            nn.Linear(128, 2)\n",
                "        )\n",
                "        \n",
                "    def forward(self, x):\n",
                "        return self.classifier(self.conv_block(x))"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# 3. Define Synthetic Data Generator
def generate_mock_data(num_samples, filename, include_label=True):
    import random
    data_list = []
    for _ in range(num_samples):
        board = [random.choice([-1, -1, -2, 0, 1, 2]) for _ in range(81)]
        hidden_indices = [i for i, val in enumerate(board) if val == -1]
        target_idx = random.choice(hidden_indices) if hidden_indices else 40
        board[target_idx] = -1
        t_row, t_col = divmod(target_idx, 9)
        board_str = " ".join(map(str, board))
        
        if include_label:
            data_list.append([board_str, t_row, t_col, random.choice([0, 1])])
        else:
            data_list.append([board_str, t_row, t_col])
            
    import pandas as pd
    cols = ['board_state', 'target_row', 'target_col', 'is_mine'] if include_label else ['board_state', 'target_row', 'target_col']
    pd.DataFrame(data_list, columns=cols).to_csv(filename, index=False)

# Write everything out to the workspace
if __name__ == "__main__":
    print("Extracting problem suite components...")
    
    # Write problem text
    with open("problem_statement.txt", "w") as f:
        f.write(problem_text)
    print("-> Created 'problem_statement.txt'")
        
    # Write Notebook JSON
    with open("starter_notebook.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    print("-> Created 'starter_notebook.ipynb'")
        
    # Generate datasets
    print("Generating simulation datasets (this may take a moment)...")
    generate_mock_data(600, "train.csv", include_label=True)
    generate_mock_data(150, "test.csv", include_label=False)
    print("-> Created 'train.csv' and 'test.csv'")
    print("\n[SUCCESS] Environment completely ready for local iteration.")