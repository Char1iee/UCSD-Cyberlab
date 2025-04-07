
# Adversarial Example Attacks on Tabular Data
The official repository of [Cost aware Feasible Attack (CaFA) on Tabular Data](TODO-LINK). It provides a modular, clean and 
accessible implementation of CaFA and its variants, complying with [Adversarial Robustness Toolbox framework](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main). 
Thus, it allows: transparency of technical details of our work, future extension of the work and utilizing the attack for practical means (e.g., evaluation of models' robustness).

<div align="center">
<img width="500" src="docs/tabular-attack-example.png">
</div>

## What is CaFA?
CaFA is an _Adversarial Example_ attack, suited for tabular data. That is, given a set of samples and a classification 
ML-model, CaFA crafts malicious inputs--based on the original ones--that are misclassified by the model.

CaFA is composed of 3 main logical components:
1. **Mine:** employing a constraints mining algorithm (we use [FastADC](https://github.com/RangerShaw/FastADC) and our ranking scheme) on a 
portion of the dataset; we focus on [Denial Constraints](https://dl.acm.org/doi/10.14778/2536258.2536262).
2. **Perturb:** attacking the model with *TabPGD* (a [PGD](https://arxiv.org/abs/1706.06083) variation we propose to attack tabular data) and *TabCWL0*
(a variation of [Carlini-Wagner](https://arxiv.org/abs/1608.04644)'s attack) to craft adversarial examples under structure constraints and cost limitations.
3. **Project:** The crafted samples are then projected onto the constrained space embodied by the constraints 
learned in the first step. For this end we use a SAT solver ([Z3 Theorem Prover](https://github.com/Z3Prover/z3)).

## Setup
The project requires `Python 3.8.5` and on, and `Java 11` and on (to run `FastADC`). Additionally, 
the installation of `pip install -r requirements.txt` is required (preferably in an isolated `venv`).

## Usage
To run the attack use:
```bash
python attack.py data=<dataset_name>
```
Where `<dataset_name>` is one of the datasets listed in the `data/` dir (which can be enriched).

The attack's components can be enabled/disabled/modified through the [Hydra](https://hydra.cc/)'s configuration dir (`config/`) or [overriden](https://hydra.cc/docs/advanced/override_grammar/basic/) through 
CLI.
These components include:
- `data`: the dataset to preprocess, train on, attack and mine constraints from.
- `ml_model`: the ML model to load/train and target as part of the attack.
- `attack`: the attack's (CaFA) parameters. 
- `constraints`: the specification of the utilized constraints, their mining process and whether to incorporate 
projection; in this these are Denial Constraints.


## Datasets
We evaluate on three commonly used tabular datasets:
[Adult](https://archive.ics.uci.edu/ml/datasets/adult) and 
[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), and
[Phishing Websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites). 

Additional tabular datasets can be added following the same structure and format as the existing ones; that is, it is requried to provide the attack with the data itself, its structure and optionally the mined constraints (see: `config/data/`). 


## Citation
If you use this code in your research, please cite our paper:
```
@inproceedings{BenTov24CaFA,
  title={{CaFA}: {C}ost-aware, Feasible Attacks With Database Constraints Against Neural Tabular Classifiers},
  author={Ben-Tov, Matan and Deutch, Daniel and Frost, Nave and Sharif, Mahmood},
  booktitle={Proceedings of the 45th IEEE Symposium on Security and Privacy (S&P)},
  year={2024}
}
```

## License
`attack-tabular` repository is licensed under the [MIT License](LICENSE).
