# BioASQ Generation Tuning Plan

## Goal

Tune the following generation-side and KG-related hyperparameters efficiently enough for a paper workflow:

1. `generation_temperature`
2. `generation_max_tokens`
3. `kg_top_k`
4. `kg_hop1_m`
5. `kg_hop2_n`
6. `kg_hop2_cap`
7. `rerank_kg_top_n`
8. `use_kg_merger`
9. `use_head_tail_placement`

The tuning workflow uses a two-stage shortlist strategy to reduce cost:

- **Stage 1**: ROUGE-SU4 only on the first 20 validation questions
- **Stage 2**: ROUGE-SU4 + full RAGAS on the first 50 validation questions

## Stage 1

### Dataset slice

- Use the **first 20 questions** from `data/val/val_bioasq.jsonl`

### Metric

- Primary metric: `ROUGE-SU4-F1`

### Search strategy

- Use **one-factor-at-a-time** search around the default baseline.
- Keep `use_ragas=False`.

### Baseline

- `generation_temperature = 0.0`
- `generation_max_tokens = 2048`
- `kg_top_k = 3`
- `kg_hop1_m = 10`
- `kg_hop2_n = 5`
- `kg_hop2_cap = 50`
- `rerank_kg_top_n = 20`
- `use_kg_merger = True`
- `use_head_tail_placement = True`

### Candidate values

- `generation_temperature`: `0.0`, `0.1`, `0.2`, `0.3`
- `generation_max_tokens`: `512`, `1024`, `2048`
- `kg_top_k`: `2`, `3`, `5`
- `kg_hop1_m`: `5`, `10`, `15`
- `kg_hop2_n`: `3`, `5`, `8`
- `kg_hop2_cap`: `30`, `50`, `80`
- `rerank_kg_top_n`: `10`, `20`, `30`
- `use_kg_merger`: `True`, `False`
- `use_head_tail_placement`: `True`, `False`

### Output

- Save each run under `results/eval_results/bioasq/generation/tuning/stage1/<run_name>/`
- Save a global `leaderboard.json`
- Save `top5_candidates.json`

## Stage 2

### Dataset slice

- Use the **first 50 questions** from `data/val/val_bioasq.jsonl`

### Metric set

- `ROUGE-SU4`
- `RAGAS Context Precision`
- `RAGAS Context Recall`
- `RAGAS Faithfulness`
- `RAGAS Answer Correctness`
- `RAGAS Answer Relevancy`

### Input

- Read `top5_candidates.json` produced by Stage 1

### Ranking rule

Sort descending by:

1. `ROUGE-SU4-F1`
2. `answer_correctness`
3. `faithfulness`
4. `answer_relevancy`

### Output

- Save each run under `results/eval_results/bioasq/generation/tuning/stage2/<run_name>/`
- Save `leaderboard.json`
- Save `best_config.json`

## Notes

- RAGAS evaluator models are configurable from `config/settings.py`
- The evaluator model is independent from the generator model
- `kg_hop1_m` keeps its current name to match the existing codebase
