# Multi-Agent Emergent Communication (MAC)

This repository provides a research framework for studying **emergent communication** in multi-agent reinforcement learning (MARL) environments. It supports a variety of on-policy algorithms and environments, enabling the exploration of how communication protocols arise, evolve, and can be interpreted in cooperative and competitive multi-agent settings.

---

## ⚠️ This fork: Causal Mechanistic Emergent Language with Online Repair

This is a **research fork**. The capstone work lives in the `mappo` → `r_mappo` stack on MPE
`simple_spread` with 2 agents — roughly 85% of the upstream repo (SMAC, Hanabi, traffic
junction, transformers) is unused here.

Two MAPPO agents learn a discrete-token communication channel. The system measures **causally**
whether messages help, breaks communication with a controlled observation perturbation, detects
the degradation, and automatically selects, applies and accepts-or-rejects an online repair.

### Read these first, in order

| doc | what it is |
|---|---|
| **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** | the authoritative design doc — what is implemented, how to run it, and all validated results (§5.9–§5.12) |
| **[TEAM_HANDOFF.md](TEAM_HANDOFF.md)** | what is done, what is still needed, in priority order — **start here if you are picking up the experiments** |
| **[REFRAME.md](REFRAME.md)** | how the original three-layer plan maps onto what was actually built |

### Quick start

```bash
conda activate marl          # required — the base env will not work
export KMP_DUPLICATE_LIB_OK=TRUE

# detect-and-repair on a trained agent (~14 min)
python onpolicy/scripts/phase2_3_repair.py   --env_name MPE --scenario_name simple_spread --algorithm_name mappo --seed 1   --n_rollout_threads 32 --n_eval_rollout_threads 1   --model_dir onpolicy/scripts/results/MPE/simple_spread/mappo/phase2_3_seed1/run1/models/checkpoint_1958400   --measure_episodes 6 --mirror_scope partner_full --repair_iters 15
```

Three trained agents and all experiment logs are committed under
`onpolicy/scripts/results/.../phase2_3_seed*/` and `results/`. See §7.1 of the overview for
which checkpoints to use — and **do not use `checkpoint_best/` without checking which step it
resolves to** (§5.12).

---

## Key Features

- **On-Policy MARL Algorithms**: Includes implementations of algorithms such as MAPPO with communication.
- **Emergent Communication**: Tools and environments for analyzing and interpreting communication between agents.
- **Diverse Environments**: Supports StarCraft II, Hanabi, MPE (Multi-Agent Particle Environments), Traffic Junction, MNIST-based tasks, and more.
- **Extensible Framework**: Modular design for easy addition of new algorithms, environments, and communication protocols.
- **Experiment Scripts**: Ready-to-use scripts for training, evaluation, and rendering across supported environments.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd MAC
   ```

2. **Install dependencies:**
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Or with conda:
     ```bash
     conda env create -f environment.yaml
     conda activate mac
     ```

## Repository Structure

- `onpolicy/algorithms/` — On-policy MARL algorithms (e.g., RMAPPo, MACPPO, MEMO_PPO)
- `onpolicy/envs/` — Supported environments (StarCraft2, Hanabi, MPE, Traffic Junction, MNIST, etc.)
- `onpolicy/runner/` — Training and evaluation runners
- `onpolicy/scripts/` — Shell scripts for launching experiments and evaluations
- `onpolicy/utils/` — Utility functions and helpers
- `onpolicy/config.py` — Centralized configuration and hyperparameter management

## Getting Started

To train a model in a supported environment, use one of the provided scripts. For example:
```bash
bash onpolicy/scripts/train/train_smac_8m.sh
```
Modify the scripts or use `onpolicy/config.py` to adjust hyperparameters and experiment settings.

## Citing This Work

If you use this repository or its components in your research, please cite the following works:

```bibtex
@article{karten2023interpretable,
  title={Interpretable learned emergent communication for human--agent teams},
  author={Karten, Seth and Tucker, Mycal and Li, Huao and Kailas, Siva and Lewis, Michael and Sycara, Katia},
  journal={IEEE Transactions on Cognitive and Developmental Systems},
  volume={15},
  number={4},
  pages={1801--1811},
  year={2023},
  publisher={IEEE}
}
@article{karten2023role,
  title={On the role of emergent communication for social learning in multi-agent reinforcement learning},
  author={Karten, Seth and Kailas, Siva and Li, Huao and Sycara, Katia},
  journal={arXiv preprint arXiv:2302.14276},
  year={2023}
}
@article{karten2022towards,
  title={Towards true lossless sparse communication in multi-agent systems},
  author={Karten, Seth and Tucker, Mycal and Kailas, Siva and Sycara, Katia},
  journal={arXiv preprint arXiv:2212.00115},
  year={2022}
}
@phdthesis{karten2023emergent,
  title={Emergent Communication and Decision-Making in Multi-Agent Teams},
  author={Karten, Seth},
  year={2023},
  school={Carnegie Mellon University Pittsburgh, PA}
}
```

## License

This project is licensed under the MIT License.
