# Selected grouped-ARD method

## Model

For layer $k$, let

$$
W^{(k)}\in\mathbb R^{n_k\times n_{k-1}}.
$$

Every scalar weight has a diagonal Gaussian posterior

$$
q(w_{ji}^{(k)})
=
\mathcal N(\mu_{ji}^{(k)},s_{ji}^{(k)2}).
$$

The prior variance is the product of layer-local input and output scales,

$$
v_{ji}^{(k)}
=
\tau_{i,\mathrm{in}}^{(k)}
\tau_{j,\mathrm{out}}^{(k)}.
$$

This makes a weight small when either its input endpoint or its output endpoint
is irrelevant. Hidden layers additionally use a two-Gaussian mixture. Its
spike variance $\xi_k$ is a learned scalar shared by the complete weight matrix:

$$
p(w_{ji}^{(k)}\mid z_{ji}^{(k)}=0)
=
\mathcal N(0,\xi_k),
$$

$$
p(w_{ji}^{(k)}\mid z_{ji}^{(k)}=1)
=
\mathcal N(0,v_{ji}^{(k)}),
$$

The spike therefore represents a single absolute noise scale rather than a
fixed fraction of every structural slab scale. The final classifier remains
deterministic and dense; columns corresponding to removed last-hidden-layer
neurons can still be deleted during compression.

## Analytical updates

Define the posterior second moment

$$
S_{ji}=\mu_{ji}^2+s_{ji}^2
$$

and spike responsibility

$$
r_{ji}=q(z_{ji}=0).
$$

The coordinate-optimal responsibility is

$$
r_{ji}
=
\operatorname{sigmoid}\left(
\operatorname{logit}\pi
+\frac12\log\frac{v_{ji}}{\xi_k}
-\frac12S_{ji}\left(\frac1{\xi_k}-\frac1{v_{ji}}\right)
\right),
$$

where $\lambda=1/\tau$. The mixture probability update is

$$
\pi=\frac1{|W|}\sum_{j,i}r_{ji}.
$$

The matrix-level spike variance also has an exact M-step:

$$
\boxed{
\xi_k
=
\frac{\sum_{j,i}r_{ji}S_{ji}}{\sum_{j,i}r_{ji}}
}.
$$

Let $u_{ji}=1-r_{ji}$ be the slab responsibility. Ignoring biases for
compactness, the structural scales occur only in the slab component, so their
alternating maximum-likelihood updates are

$$
\lambda_{j,\mathrm{out}}
=
\frac{\sum_i u_{ji}}{
\sum_i
u_{ji}S_{ji}\lambda_{i,\mathrm{in}}
},
$$

$$
\lambda_{i,\mathrm{in}}
=
\frac{\sum_j u_{ji}}{
\sum_j
u_{ji}S_{ji}\lambda_{j,\mathrm{out}}
}.
$$

For an output scale whose Gaussian bias belongs to the slab, add one to its
numerator and add the bias second moment to its denominator. These coordinate
updates are performed every optimization iteration. A small numerical floor
on $\xi_k$ prevents the standard singularity of unconstrained Gaussian-mixture
maximum likelihood. Gradients update $\mu$ and $\log s^2$ using the
reparameterized expected negative log likelihood plus the Gaussian-mixture
variational KL. The KL coefficient is turned on gradually after deterministic
pretraining.

## Neuron importance

The retained parameter-based importance is the slab-conditioned row SNR. Let

$$
u_{ji}=1-r_{ji}.
$$

For output neuron $j$,

$$
\boxed{
I_j
=
\frac{
\mu_{b,j}^2+
\sum_i u_{ji}\mu_{ji}^2
}{
s_{b,j}^2+
\sum_i u_{ji}s_{ji}^2
}.
}
$$

The sums are taken before their ratio. Effective slab support

$$
M_j=\sum_i u_{ji}
$$

is reported separately and is not multiplied into $I_j$. A numerical pruning
threshold must be validated by constructing or masking the corresponding
smaller dense network and measuring its validation loss.

## Weight-level posterior SNR diagnostic

For every matrix weight, define

$$
\ell_{ji}
=
\log\frac{\mu_{ji}^2}{s_{ji}^2}
=
2\log|\mu_{ji}|-\log s_{ji}^2.
$$

This diagnostic is not a neuron-pruning score: it deliberately ignores the
mixture responsibility and shared neuron structure. Its histogram shows
whether individual posterior weights separate into low- and high-SNR modes or
remain spread through an ambiguous intermediate regime. In code, $\mu_{ji}^2$
is clamped only to the smallest positive number representable by its dtype so
that an exactly zero mean maps to a finite left-tail value.

## Selected experiment

The selected learned shared-spike-variance experiment fine-tunes the $98.39\%$
deterministic MNIST baseline with a $784$-$300$-$100$-$10$ architecture, dense
final classifier, learning rate $10^{-4}$, 10 zero-KL epochs, a 60-epoch linear
KL ramp, and 30 full-KL epochs. Each $\xi_k$ is initialized to $10^{-4}$ and
then receives the exact update above.

- Best full-KL accuracy: $97.66\%$ at epoch 70.
- Final accuracy: $97.61\%$.
- Final KL per training example: $0.4615$.
- Final spike variances: approximately $2.76\times10^{-5}$ for the first
  matrix and $2.60\times10^{-5}$ for the second matrix.

The learned-variance reference artifacts are in
`artifacts/mnist_ard_learned_spike_variance_dense_final/`.
