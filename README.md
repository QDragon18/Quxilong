Hypergraph Influence Maximization
This repository contains the open-source code for the paper Evolutionary Hypergraph Learning: A Multi-Objective Influence Maximization Approach Considering Seed Selection Costs. The code implements algorithms for influence maximization on hypergraphs, optimizing both influence spread and seed selection cost.
Overview
The run-experiment.py script implements heuristic algorithms (Neighbor Priority, Hyper Degree, HCI1, HCI2, PageRank, Random) and evolutionary algorithms (GA, NSGA-II, NSGA-II-Init, MOEA-IMF) to select seed nodes in a hypergraph. It uses Monte Carlo simulations to evaluate influence spread and supports both constant and degree-based activation probabilities.

Author: Qu Xilong (First Author)
Contact: 1244559411@qq.com

Usage
Run the main experiment with the provided script:
python run-experiment.py

The script processes a hypergraph from ./test/Restaurant-rev.txt and outputs results to ./test/:

spread_results-<i>.csv: Pareto front results for simulation i.
result_detail-<i>.csv: Generation-wise results for evolutionary algorithms.
spread_results_others.csv: Results from heuristic algorithms.
random_seeds.txt: Random seeds for reproducibility.
diversity.csv: Diversity metrics for MOEA-IMF.

To customize, modify the main() function in run-experiment.py with your hypergraph file and parameters (e.g., node_size, edge_size, identical_p).
Dependencies

Python 3.x
Libraries: deap, numpy, pandas, matplotlib, hypernetx, tqdm

Install via:
pip install -r requirements.txt

File Structure
hypergraph-influence-maximization/
├── run-experiment.py       # Main script with all algorithms
├── test/
│   ├── Restaurant-rev.txt  # Sample hypergraph data
│   ├── spread_results-*.csv
│   ├── result_detail-*.csv
│   ├── spread_results_others.csv
│   ├── random_seeds.txt
│   ├── diversity.csv
├── README.md               # This file
