# Long cluster run

The cluster entry point trains a fresh deterministic baseline and then the
selected two-sided grouped-ARD model with a fixed two-Gaussian variance ratio.
The final classifier stays dense.

## Default schedule

The ARD objective is

$$
\mathcal L_t
=
\mathbb E_q[-\log p(\mathcal D\mid W)]
+\beta_t\operatorname{KL}(q(W)\Vert p(W)),
$$

with

$$
\beta_t
=
\begin{cases}
0, & t\leq 30,\\
\min\left(1,\dfrac{t-30}{200}\right), & t>30.
\end{cases}
$$

The defaults are:

- Baseline: 100 epochs, learning rate $10^{-3}$.
- ARD: 300 epochs, learning rate $10^{-4}$.
- KL schedule: 30 epochs off, 200-epoch ramp, 70 epochs at full strength.
- Mixture ratio: $\kappa=10^{-2}$.
- Architecture: $784$-$300$-$100$-$10$.
- Checkpoints: every 10 ARD epochs.
- Selection: retain the highest-accuracy checkpoint after $\beta_t=1$.
- Device selection: CUDA, then MPS, then CPU.

## Run

From the repository root:

```bash
uv sync
scripts/run_long_mnist.sh
```

The script downloads MNIST when necessary and writes everything beneath
`artifacts/mnist_long/`, which Git ignores. The selected checkpoint is
`ard_fixed_ratio/best_full_kl_model.pt`; importance plots are produced for
both it and the final model.

Common cluster overrides:

```bash
DEVICE=cuda \
NUM_WORKERS=8 \
BATCH_SIZE=512 \
OUTPUT_ROOT=/path/to/persistent/results/mnist_long \
scripts/run_long_mnist.sh
```

To reuse an existing deterministic checkpoint and skip baseline training:

```bash
PRETRAINED_CHECKPOINT_OVERRIDE=/path/to/best_model.pt \
scripts/run_long_mnist.sh
```

Every setting in the script can be overridden through its same-named
environment variable. In particular, `ARD_EPOCHS`, `KL_ZERO_EPOCHS`, and
`KL_WARMUP_EPOCHS` control the schedule.
