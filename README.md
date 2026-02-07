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

