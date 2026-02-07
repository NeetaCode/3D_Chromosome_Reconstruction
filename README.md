# 3D Coordinate Alignment and PDB Export

End-to-end pipeline to **align predicted 3D coordinates** to **ground-truth coordinates**, evaluate alignment quality with **Pearson/Spearman**, and export the adjusted structure to **PDB**. Includes an experimental **RL (DQN)** prototype for coordinate adjustment plus graph utilities using **PyTorch Geometric**.

---

## What this does

Given two files with matching rows of `(x, y, z)` points:

- **Actual / reference coordinates** (ground truth)
- **Predicted coordinates** (model output)

This project provides:

1. **Supervised calibration (recommended)**
   - Trains a small MLP to learn `predicted_xyz → actual_xyz`
   - Produces **adjusted coordinates**
2. **Evaluation**
   - Pearson correlation (linear agreement)
   - Spearman correlation (rank agreement)
3. **Exports**
   - Saves adjusted coordinates to `.txt`
   - Converts coordinates to `.pdb` for visualization
4. **Experimental RL approach (prototype)**
   - DQN-style agent tries to learn coordinate adjustments
   - Included for exploration, not currently competitive
5. **Graph utilities**
   - Load adjacency/edge-list formats into a PyG `Data` object
   - Helper functions for contact→distance conversion and domain alignment

---

## Data format

### Coordinate files
Text files with **N rows** and **3 columns**:

x y z
x y z
...

Examples shown in the notebook:
- Actual coordinates have large magnitudes (e.g. `-2368, -986, 321`)
- Predicted coordinates are smaller scale (e.g. `4.6, -8.1, 0.36`)

This mismatch is expected. The supervised calibration learns the mapping.

---

## Dependencies

### Core
- `numpy`, `pandas`, `scipy`
- `tensorflow` (Keras)

### RL / Gym
- `gymnasium` / `gym`
- `shimmy>=0.2.1`
- `stable-baselines3` (installed; PPO imported but the final RL prototype uses a custom DQN)

### Graph utilities
- `torch`
- `torch-geometric`
- `networkx`
- `torch_sparse`, `torch_scatter`

---

## Install

Colab-friendly installs:

```bash
pip install stable-baselines3
pip install "shimmy>=0.2.1"

pip install scipy
pip install torch-geometric
pip install torch_sparse
pip install torch_scatter


Recommended workflow (supervised calibration)
1) Load coordinates

import numpy as np

actual = np.loadtxt("path/to/actual.txt")
pred   = np.loadtxt("path/to/predicted.txt")

# Must be (N, 3) and same N
assert actual.shape == pred.shape
assert actual.shape[1] == 3

2) Train MLP mapping predicted → actual (MSE)
Model:
•	Dense(64, ReLU)
•	Dense(64, ReLU)
•	Dense(3)
Loss: MSE
Optimizer: Adam

import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(3,)):
    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dense(3),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = create_model((pred.shape[1],))
model.fit(pred, actual, epochs=1000, verbose=0)

adjusted = model.predict(pred)

3) Evaluate Pearson and Spearman
Flattened evaluation compares all coordinates jointly:
import pandas as pd
from scipy.stats import pearsonr, spearmanr

pearson  = pearsonr(actual.flatten(), adjusted.flatten())[0]
spearman = spearmanr(actual.flatten(), adjusted.flatten())[0]

print("Pearson:", pearson)
print("Spearman:", spearman)
Observed in your run (example):
•	Initial: Pearson ~0.13, Spearman ~0.08
•	After supervised calibration: Pearson ~0.9995, Spearman ~0.9940
________________________________________
RL prototype (DQN-style coordinate adjustment)
This repo includes a simple environment + DQN agent as an experiment.
Environment definition
•	State: coordinate at index i (shape (3,))
•	Action: delta (dx, dy, dz) (continuous vector)
•	Transition: move to the next index each step
•	Reward variants tested:
1.	-||adjusted - actual|| (negative L2)
2.	-1 / (1 + ||adjusted - actual||) (bounded negative)
Agent definition
•	Replay buffer (deque)
•	Epsilon-greedy exploration using uniform random actions in [-0.1, 0.1]
•	MLP outputs 3 values used as the action vector
Important limitation
This is labeled “DQN”, but classic DQN is for discrete actions. The current update pattern uses np.argmax(action) on a continuous action vector, which does not represent a correct Q-learning update for continuous control.
Empirical outcome in your runs:
•	RL correlations remained low (Pearson ~0.03 to ~0.13, Spearman ~0.08 to ~0.09)
If RL is required, continuous-control algorithms like DDPG / TD3 / SAC are the correct next step.
________________________________________
Saving adjusted coordinates
Save .txt:
import numpy as np

np.savetxt(
    "adjusted_coordinates.txt",
    adjusted,
    delimiter="\t"
)
PDB export
Two writers are included.
A) Simple TXT → PDB
Reads a tab-separated coordinate text file and writes a minimal PDB:
•	HEADER
•	ATOM records
•	TER
•	END
def convert_to_pdb(txt_file_path, pdb_file_path):
    with open(txt_file_path, "r") as f:
        lines = f.readlines()

    with open(pdb_file_path, "w") as f:
        f.write("HEADER    Generated by Python\n")
        atom_count = 0
        for line in lines:
            atom_count += 1
            x, y, z = line.strip().split("\t")
            f.write(f"ATOM  {atom_count:5d}  C   UNK A{atom_count:4d}    {x:8s}{y:8s}{z:8s}\n")
        f.write("TER\n")
        f.write("END\n")
B) WritePDB(positions, pdb_file, ctype="0")
Writes:
•	ATOM entries with formatted coordinates
•	CONECT entries linking sequential atoms (chain-like)
________________________________________
Graph utilities (PyTorch Geometric)
These utilities help load adjacency-like input into graphs.
Supported input formats
•	Dense adjacency matrix
•	Edge list / contact list format with 3 columns (i, j, weight) converted to dense matrix
Key functions
•	convert_to_matrix(adj): edge list → symmetric dense adjacency matrix
•	load_input(input_file, features):
o	loads adjacency or edge list
o	builds NetworkX graph, then PyG Data(x, edge_index, y)
o	creates symmetric SparseTensor adjacency
•	cont2dist(adj, factor):
o	converts contact strengths to normalized distances via (1/adj)**factor
•	domain_alignment(...):
o	orthogonal Procrustes alignment between embedding spaces


