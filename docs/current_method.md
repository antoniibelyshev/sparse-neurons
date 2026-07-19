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

Bias is represented by the augmented constant input $x_0=1$ with
$w_{j0}=b_j$. Its activation is fixed, but its prior input scale
$\tau_{0,\mathrm{in}}$ is learned like the other entries. Consequently,

$$
v_{j0}^{(k)}
=
\tau_{0,\mathrm{in}}^{(k)}\tau_{j,\mathrm{out}}^{(k)}.
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

The spike therefore represents a single absolute noise scale. The final
classifier remains deterministic and dense; columns corresponding to removed
last-hidden-layer neurons can still be deleted during compression.

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
\pi
=
\frac1{n_k(n_{k-1}+1)}
\sum_j\sum_{i=0}^{n_{k-1}}r_{ji}.
$$

The matrix-level spike variance also has an exact M-step:

$$
\boxed{
\xi_k
=
\frac{
\sum_j\sum_{i=0}^{n_{k-1}}r_{ji}S_{ji}
}{
\sum_j\sum_{i=0}^{n_{k-1}}r_{ji}
}
}.
$$

Let $u_{ji}=1-r_{ji}$ be the slab responsibility. The structural scales occur
only in the slab component, so their alternating maximum-likelihood updates
are

$$
\lambda_{j,\mathrm{out}}
=
\frac{\sum_{i=0}^{n_{k-1}} u_{ji}}{
\sum_{i=0}^{n_{k-1}}
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

The input-scale update includes $i=0$, so the bias column has one learned prior
scale shared across all output neurons. These coordinate updates are performed
every optimization iteration. A small numerical floor on $\xi_k$ prevents the
standard singularity of unconstrained Gaussian-mixture maximum likelihood.
Gradients update $\mu$ and $\log s^2$ using the reparameterized expected
negative log likelihood plus the Gaussian-mixture variational KL. The KL
coefficient is turned on gradually after deterministic pretraining.

## Neuron importance

Introduce the constant augmented input $x_0=1$ and represent the bias as
$w_{j0}=b_j$. The parameter-based importance of output neuron $j$ is the
largest posterior SNR among all its augmented incoming weights:

$$
\boxed{
I_j
=
\max_{i\in\{0,\ldots,n_{\mathrm{in}}\}}
\frac{\mu_{ji}^2}{s_{ji}^2}.
}
$$

This makes the bias contribution identical to that of any other connection,
with its input activation fixed at one. The score keeps a neuron whenever it
has at least one incoming connection whose posterior mean is large relative
to its uncertainty. It is deliberately conservative: every augmented weight
must have low SNR before the neuron receives low importance. A numerical
pruning threshold must still be validated by constructing the corresponding
smaller dense network and measuring its validation loss.

## Weight-level posterior SNR diagnostic

For every augmented weight, including $w_{j0}=b_j$, define

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
