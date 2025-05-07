# GNN

This submodule contains the Graph Neural Network training pipelines and logs for LLVM IR-based graph classification.

## Files

- **GNN_training.py**  
  Defines the GNN architecture and training logic for classifying LLVM IR graphs.

- **GNN1_training.py**  
  Implements an advanced GAT-LSTM hybrid with mixed precision and gradient clipping.

- **graph_generate.py**  
  Loads CSV-based graph data and converts it to PyTorch Geometric `Data` objects.

- **data_check.py**  
  Utility to inspect and summarize graph datasets.

- **script.sh**  
  Script to run `GNN1_training.py` in the background and log output.

## Logs

- Multiple `training_log_*.txt` files record warnings, training metrics, and timestamps for tracking experiments.
