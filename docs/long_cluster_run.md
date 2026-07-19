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
\begin{cases}
0, & t\leq 30,\\
\min\left(1,\dfrac{t-30}{200}\right), & t>30.
\end{cases}
$$

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

The deterministic baseline minimizes

$$
\mathcal L_{\mathrm{base}}(\theta)
=
-\frac1N\sum_{n=1}^N\log p(y_n\mid x_n,\theta)
+\frac{\gamma}{2}\lVert\theta\rVert_2^2,
$$

with $\gamma=10^{-4}$. This mild baseline-only penalty discourages a sharp,
large-weight interpolating solution before variational fine-tuning. It does
not add another penalty to the ARD phase, whose Gaussian-mixture KL already
regularizes its weights.

The defaults are:

- Baseline: 100 epochs, learning rate $10^{-3}$, L2 coefficient $10^{-4}$.
- ARD: 300 epochs, initial learning rate $10^{-3}$ with cosine decay.
- Batch size: 1024 for both phases.
- KL schedule: 30 epochs off, 200-epoch ramp, 70 epochs at full strength.
- Initial low-mode variance: $\xi_k=10^{-4}$, followed by exact M-steps.
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

On Linux, `uv sync` installs the official PyTorch 2.6 CUDA 12.4 build. This is
compatible with NVIDIA drivers reporting CUDA 12.4 and avoids accidentally
resolving a newer CUDA runtime than the cluster driver supports.

The script downloads MNIST when necessary and writes everything beneath
`artifacts/mnist_long/`, which Git ignores. The selected checkpoint is
`ard_learned_spike_variance/best_full_kl_model.pt`; importance plots are produced for
both it and the final model.

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
