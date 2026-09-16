# LLM Optimization with Block-wise SVD and a Genetic Algorithm

Compressing a fine-tuned transformer by searching, with a genetic algorithm, for the best per-block low-rank factorization of its least sensitive linear layers.

**Marcel Yammine · Anthony Yang · François Zapletal** — DIA 6, ESILV
Course: *AI Algorithms* (MESIIN476025) — Final Project, Option A

---

## Problem

Large language models are expensive to store and serve, and model compression is one of the levers used to make them deployable. Singular Value Decomposition (SVD) is a classic way to approximate a weight matrix with a lower-rank one, but it raises an immediate question: **which rank, for which layer?**

Applying a single global rank is crude. Some layers tolerate heavy truncation without any measurable loss, others collapse the model. Even inside one weight matrix, different regions carry different amounts of information.

This project treats rank selection as a **combinatorial optimization problem** and solves it with a genetic algorithm. Each weight matrix is split into blocks, every block gets its own rank, and the GA searches the resulting space for the configuration that gives the best accuracy-versus-size trade-off.

## Approach

**1. Baseline**
`distilbert-base-uncased-finetuned-sst-2-english` evaluated on the GLUE SST-2 validation set. Accuracy and model size in MB are measured and used as reference points for everything that follows.

**2. Layer sensitivity analysis**
Every `nn.Linear` layer is perturbed one at a time with Gaussian noise (ε = 0.1) and the model is re-evaluated. The accuracy drop gives a sensitivity score per layer. The least sensitive layers are the safest candidates for compression, which keeps the search space small and avoids wasting generations on layers that would break the model.

**3. Block-wise SVD**
A weight matrix is tiled into a grid of blocks whose shape is derived from the requested number of blocks and the matrix aspect ratio. Each block is factorized independently at its own rank. A custom `BlockLowRankLinear` module stores the `(U, S, V)` triplets and reconstructs the matrix at forward time, so the compressed layer is a drop-in replacement for the original `nn.Linear`.

**4. Genetic algorithm**

| Component | Design |
|---|---|
| Chromosome | one gene per selected layer: `{nb_blocks, ranks[]}` |
| Fitness | `accuracy / baseline_accuracy − λ · (size / baseline_size)`, with λ = 0.5 |
| Selection | elitism, top 2 carried over unchanged |
| Crossover | single point, over the ordered list of layers |
| Mutation | resample `nb_blocks` (which regenerates the rank vector), or perturb individual ranks, at rate 0.2 |
| Search space | `nb_blocks ∈ {4, 8, 16}`, `rank ∈ [1, 32]` |
| Run | population 10, 5 generations |

The fitness function is normalized against the baseline so that accuracy and size are comparable, and λ sets how aggressively the search trades one for the other.

An important methodological point: the GA is scored on a **held-out split** (`validation[200:600]`) while final results are reported on `validation[:200]`. Optimizing and reporting on the same data would only measure how well the GA overfits its own fitness set.

## Results

Five least sensitive layers selected, mostly in the last transformer block plus the classifier head.

| | Accuracy | Size | Compression |
|---|---|---|---|
| Baseline | 0.9100 | 255.41 MB | 1.00× |
| Optimized | 0.9050 | 234.76 MB | 1.09× |

Roughly 8 % of the model size removed for a 0.55 % accuracy drop.

**Study across block granularity** — the GA was re-run with the number of blocks fixed to 4, 8 and 16. At 16 blocks the compressed model reached **0.9200 accuracy, above the baseline**, at a comparable size. Finer blocks give the algorithm more degrees of freedom: a block that carries little information can be truncated hard while an informative one keeps a high rank, which a single per-layer rank cannot express. The accuracy going slightly above baseline is most likely a regularization effect combined with the small size of the evaluation split, not a genuine improvement in model quality.

## Limitations

Worth stating plainly, since they shape how the numbers should be read.

- **The compression ratio is modest (1.09×)** because only five layers are compressed. Extending the selection would increase the gain, at the cost of a larger search space and a higher risk of accuracy loss.
- **`BlockLowRankLinear` rebuilds the full matrix on every forward pass.** Storage is reduced, inference cost is not. A production implementation would keep the computation factorized as `x @ Vᵀ @ diag(S) @ Uᵀ`.
- **Evaluation splits are small** (200–600 examples), which makes accuracy differences of a few tenths of a point noisy.
- **The GA plateaus quickly** — best fitness barely moves after generation 2. A larger population, more generations, or a less greedy selection scheme would be the first things to try.

## Running it

```bash
pip install transformers datasets accelerate bitsandbytes scipy
```

Then open `AI_Algo_Project_Final.ipynb` and run the cells in order. A GPU is not required but the sensitivity analysis and the GA loop are evaluation-heavy: each chromosome means a full deep copy of the model plus a pass over the fitness set.

Main dependencies: `torch`, `transformers`, `datasets`, `numpy`, `matplotlib`, `seaborn`, `pandas`.

## Repository

```
AI_Algo_Project_Final.ipynb   Full pipeline: baseline, sensitivity, block SVD, GA, study
README.md
```

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Hsu et al., *Language Model Compression with Weighted Low-Rank Factorization* (2022)
- Wang et al., *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding* (2018)
