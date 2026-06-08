# KDMC: Integrated Knowledge-Data Framework for Causal Discovery with Missing Data

**KDMC** is a PyTorch implementation of a knowledge-data integrated framework
for causal discovery from incomplete tabular data. The method combines
LLM-based missingness reasoning, triplet-level causal prior construction,
hierarchical mixture prior fusion, graph-attention reinforcement learning, and
BIC-based DAG optimization.

**Environment:** Python 3.10.18 | PyTorch 2.0.1 | OpenAI-compatible API required

This repository provides the core KDMC implementation, prompt-generation
modules, ASI-MCAR example data, and scripts for running a reproducible
LLM-assisted causal discovery case.

## Overview

KDMC follows the pipeline described in the paper:

<div align="center">
  <img src="docs/kdmc_framework.png" alt="KDMC framework overview" width="100%">
</div>


1. **Knowledge-driven causal reasoning**: compute mask-based missingness
   statistics, generate a structured missingness report, and query an LLM for
   triplet-level local causal relations.
2. **Hierarchical mixture prior**: aggregate LLM triplet outputs into a
   conflict-resolved prior graph, then split candidate edges into hard-prior,
   soft-prior, and free regions according to prior confidence and missing rate.
3. **Data-driven graph optimization**: use a graph-attention actor, order
   decoder, critic network, and BIC-based local scoring to optimize a DAG from
   incomplete data.
4. **Differential feedback**: summarize structural differences between the LLM
   prior and the optimized graph for subsequent reasoning rounds.

All runnable experiments in this release require API access because KDMC builds
the causal prior through LLM queries.

## Key Features

- **Missingness report generation**: summarizes variable-level missing rates,
  conditional missingness, missingness patterns, and bias risks.
- **Triplet-level LLM prior construction**: queries local three-variable DAGs
  instead of asking for a full graph directly.
- **Conflict-resolved prior aggregation**: converts repeated local LLM judgments
  into a weighted acyclic prior graph.
- **Hierarchical mixture prior strategy**: treats high-confidence prior edges as
  hard guidance, lower-confidence edges as soft guidance, and unsupported edges
  as free search space.
- **GAT-based reinforcement learning**: uses graph attention, an order decoder,
  and actor-critic optimization for DAG search.
- **BIC-based structure scoring**: evaluates candidate parent sets using
  data-driven local scores under missing observations.

## Repository Structure

```text
KDMC-open-source/
+-- main.py                         # RL-based structure optimization
+-- run_all.py                      # Missingness report + LLM prior + RL pipeline
+-- requirements.txt                # Runtime dependencies
+-- .env.example                    # API environment template
+-- docs/
|   +-- kdmc_framework.png          # Framework overview figure
+-- model/
|   +-- networks.py                 # GAT actor, order decoder, and critic
|   +-- scoring.py                  # BIC structure scoring
|   +-- structure_learner.py        # KDMC graph optimization logic
|   +-- training.py                 # Training recorder
+-- prompt_generate/                # Dataset templates and LLM prior prompts
|   +-- miss_report/                # Missingness statistics and report generation
+-- realdata/
|   +-- miss_data/ASI/              # ASI MCAR data at rates 0.2/0.4/0.6/0.8
|   +-- raw_data/ASI/               # ASI reference graph
+-- scripts/
    +-- check_api.py                # API connectivity check
```

## Quick Start

### Installation

```bash
conda create -n kdmc python=3.10.18
conda activate kdmc
pip install -r requirements.txt
```

### API Configuration

Create `.env` from `.env.example`, then fill in your OpenAI-compatible API
settings:

```text
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_openai_compatible_endpoint
OPENAI_MODEL=gpt-5.4
```

Check the connection:

```bash
python scripts/check_api.py
```

### Run One ASI-MCAR Case

The following command runs one compact ASI-MCAR case, including missingness
report generation, LLM prior construction, and RL-based graph optimization:

```bash
python run_all.py \
  --datapath realdata/miss_data/ASI/ASI_MCAR_0.2.csv \
  --labelpath realdata/raw_data/ASI/ASI_label.csv \
  --template ASI \
  --llm_model gpt-5.4 \
  --missing_rate 0.2
```

## Main Arguments

| Argument | Description |
|---|---|
| `datapath` | Path to the incomplete data CSV file. |
| `labelpath` | Path to the reference DAG CSV file for evaluation. |
| `template` | Dataset prompt template, e.g. `ASI`. |
| `llm_model` | LLM model name used by the OpenAI-compatible API. |
| `missing_rate` | Missing rate used by the adaptive prior threshold. |
| `iterations` | Number of LLM-RL differential-feedback rounds. |
| `epoch` | Number of RL training epochs in each round. |
| `prior_strategy` | Prior fusion strategy, e.g. `adaptive_hmp`. |
| `confidence_weight` | Weight for injecting LLM confidence into GAT attention. |
| `triple_chunk_size` | Number of triplets sent to the LLM per request. |
| `nheads` | Number of GAT attention heads. |
| `nblocks` | Number of GAT blocks. |
| `batch_size` | RL training batch size. |

## Outputs

Each run saves intermediate missingness reports, LLM prior files, differential
feedback, learned graphs, and evaluation metrics under `outputs/` or the
configured result directory. Reported graph metrics include SHD, F1, precision,
recall, TPR, FDR, and related diagnostic scores.

## Acknowledgement

The RL-based causal structure optimization code follows the actor-critic
structure search design of [OuTingYun/COKE](https://github.com/OuTingYun/COKE).
We thank the authors for their contribution.

## Citation

If you find this repository useful, please cite the corresponding paper once it
is released.
