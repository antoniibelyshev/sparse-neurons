# Long cluster run

The cluster entry point trains a fresh deterministic baseline and then the
selected two-sided grouped-ARD model with a two-Gaussian prior.
The low-covariance Gaussian has one trainable variance $\xi_k$ per weight
matrix. The final classifier stays dense.

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
\min\left(1,\frac{t}{200}\right).
$$

There is no KL-free phase. The copied posterior is initialized with

$$
s_{ji}=\max\left(10^{-2}|\mu_{ji}|,10^{-6}\right),
$$

and training optimizes $\log s_{ji}$ directly.

Both training phases use cosine learning-rate decay. For a phase of $T$
epochs,

$$
\eta_t
=
\eta_{\min}
+\frac12(\eta_0-\eta_{\min})
\left(1+\cos\frac{\pi t}{T}\right),
$$

where $\eta_0=10^{-3}$ and $\eta_{\min}=10^{-5}$.

Both phases maintain an exponential moving average of trainable parameters,

$$
\bar\theta_t
=
\rho\bar\theta_{t-1}+(1-\rho)\theta_t,
\qquad \rho=0.999,
$$

initialized with $\bar\theta_0=\theta_0$. Evaluation uses $\bar\theta_t$, and
the final epoch is retained without selecting the maximum test accuracy. For
ARD, the analytical mixture responsibilities and scale buffers are recomputed
after copying the EMA parameters. ARD starts from the baseline checkpoint
`final_ema_model.pt`.

The deterministic baseline minimizes

$$
\mathcal L_{\mathrm{base}}(\theta)
=
-\frac1N\sum_{n=1}^N\log p(y_n\mid x_n,\theta)
+\frac{\gamma}{2}\lVert\theta\rVert_2^2,
$$

with $\gamma=10^{-3}$. This baseline-only penalty discourages a sharp,
large-weight interpolating solution before variational fine-tuning. It does
not add another penalty to the ARD phase, whose Gaussian-mixture KL already
regularizes its weights.

The defaults are:

- Baseline: 100 epochs, learning rate $10^{-3}$, L2 coefficient $10^{-3}$.
- ARD: 300 epochs, initial learning rate $10^{-3}$ with cosine decay.
- Batch size: 1024 for both phases.
- KL schedule: 200-epoch ramp from the first epoch, then 100 full-KL epochs.
- EMA decay: $0.999$ in both phases.
- Initial low-mode variance: $\xi_k=10^{-4}$, followed by exact M-steps.
- Architecture: $784$-$300$-$100$-$10$.
- Checkpoints: every 10 ARD epochs.
- Selection: use the final EMA checkpoint for both phases.
- Device selection: CUDA, then MPS, then CPU.

## Run

From the repository root:

```bash
uv sync
scripts/run_long_mnist.sh
```

On Linux, `uv sync` installs the official PyTorch 2.6 CUDA 12.4 build. This is
compatible with NVIDIA drivers reporting CUDA 12.4 and avoids accidentally
resolving a newer CUDA runtime than the cluster driver supports.

The script downloads MNIST when necessary and writes everything beneath
`artifacts/mnist_long/`, which Git ignores. The final ARD checkpoint is
`ard_learned_spike_variance/model.pt`; its importance plots are generated
automatically. Evaluation also writes `weight_kl_diagnostics.png` and
`top_weight_kl_contributors.csv`, containing the exact per-weight mixture-KL
decomposition. At startup, the script replaces its `baseline/` and
`ard_learned_spike_variance/` output directories so obsolete checkpoints from
an earlier formulation cannot coexist with the new run.

Common cluster overrides:

```bash
DEVICE=cuda \
NUM_WORKERS=8 \
BATCH_SIZE=1024 \
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
`KL_WARMUP_EPOCHS` control the schedule. `BASELINE_WEIGHT_DECAY=0` disables
baseline L2 regularization.

## Plot an existing run

To generate diagnostics without retraining:

```bash
scripts/plot_trained_run.sh
```

For a checkpoint in another location:

```bash
DEVICE=cuda scripts/plot_trained_run.sh /path/to/model.pt /path/to/plots
```
