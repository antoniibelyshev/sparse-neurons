# CIFAR-10 VGG-11 channel-ARD experiment

## Architecture

The deterministic baseline is VGG-11 without batch normalization, adapted to
$32\times32$ CIFAR-10 images. Five max-pooling stages reduce the final
$512\times1\times1$ feature tensor to a dense ten-class classifier. The final
classifier remains deterministic; all eight convolutions become variational
channel-ARD layers.

## Convolutional prior

Let

$$
W^{(k)}\in
\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times H\times W}.
$$

For output channel $j$, input channel $i$, and spatial kernel position $p$,

$$
q(w_{jip})=\mathcal N(\mu_{jip},s_{jip}^2),
$$

and the slab variance is

$$
v_{ji}
=
\tau_{j,\mathrm{out}}\tau_{i,\mathrm{in}}.
$$

All $H W$ scalar kernel weights connecting the same pair of channels share
this structural scale but retain independent posterior parameters and mixture
responsibilities. Bias is appended as one augmented scalar per output channel,
with a learned augmented input scale.

With $S_{jip}=\mu_{jip}^2+s_{jip}^2$ and slab responsibility
$u_{jip}=1-r_{jip}$, the output-channel precision update is

$$
\boxed{
\lambda_{j,\mathrm{out}}
=
\frac{
\sum_{i,p}u_{jip}
}{
\sum_{i,p}u_{jip}S_{jip}\lambda_{i,\mathrm{in}}
}.
}
$$

The input-channel update aggregates over output channels and kernel positions:

$$
\boxed{
\lambda_{i,\mathrm{in}}
=
\frac{
\sum_{j,p}u_{jip}
}{
\sum_{j,p}u_{jip}S_{jip}\lambda_{j,\mathrm{out}}
}.
}
$$

The bias column uses the same formulas with a single position rather than
$H W$ positions. Mixture responsibilities, the learned low-mode variance
$\xi_k$, and the elementwise mixture KL are identical to the linear-layer
derivation in [current_method.md](current_method.md).

## Channel importance

Output-channel importance is the maximum scalar posterior SNR over the complete
flattened kernel row and its bias:

$$
\boxed{
I_j
=
\max_{i,p}\frac{\mu_{jip}^2}{s_{jip}^2}.
}
$$

This conservative score marks a channel unimportant only when every incoming
kernel weight and its bias have low posterior SNR.

## Training schedule

- Baseline: 100 epochs, Adam, EMA decay $0.999$, L2 coefficient $5\times10^{-4}$.
- ARD: 300 epochs, Adam, EMA decay $0.999$, no optimizer weight decay.
- Both learning rates decay from $10^{-3}$ to $10^{-5}$ with a cosine schedule.
- Batch size: 256.
- KL coefficient: $\beta_t=\min(1,t/200)$, giving 200 ramp epochs and 100
  full-KL epochs.
- Posterior initialization: $s=\max(10^{-2}|\mu|,10^{-6})$.
- Final EMA checkpoints are used without best-epoch selection.

Run from the repository root:

```bash
uv sync
DEVICE=cuda scripts/run_cifar10_vgg.sh
```

Outputs are written below `artifacts/cifar10_vgg/`, which is replaced at the
start of each run and ignored by Git.
