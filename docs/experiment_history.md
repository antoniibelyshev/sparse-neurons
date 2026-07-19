# Compressed experiment history

Only the selected run's artifacts are retained. The entries below preserve the
ideas and headline outcomes of deleted experiments.

| Variant | Brief idea | Result |
|---|---|---|
| Direct row ARD from scratch | One Gaussian precision shared by each incoming row. | About $92.6\%$; substantially below the deterministic model. |
| Deterministic pretraining | Train dense LeNet-$300$-$100$ before introducing uncertainty. | $98.39\%$ best baseline; essential for later accuracy. |
| Immediate pretrained ARD | Convert the baseline and enable KL quickly. | Collapsed toward $94$-$96\%$. |
| KL warm-up | Learn the data fit first, then increase the KL coefficient. | Improved stability and accuracy; slower ramps worked better. |
| Two-sided scales | Use input and output scale vectors so either endpoint can suppress a weight. | Fixed the width-consistency problem conceptually, but early schedules reached only about $96\%$. |
| Fixed final output scale | Remove one scale ambiguity by fixing the final variational layer's output scale. | Little accuracy improvement. |
| Relative variance initialization | Initialize $s^2$ proportional to $\mu^2$. | Ratio $10^{-2}$ reached about $97.3\%$ full-KL; smaller ratios were much worse. |
| Longer training and learning-rate decay | Extend optimization after the KL ramp. | Did not recover the lost accuracy; about $95.1\%$ best full-KL. |
| Two-Gaussian scalar mixture | Add narrow spike and broad slab components inside the two-sided scale model. | Improved best full-KL accuracy to $98.00\%$. |
| Learned matrix-level spike variance | Use one absolute $\xi_k$ per matrix with an exact M-step. | Best full-KL $97.66\%$, final $97.61\%$, KL/example $0.4615$; selected mixture formulation. |
| L2-regularized baseline | Train the deterministic initializer with coefficient $10^{-4}$. | Improved accuracy from $98.39\%$ to $98.49\%$, test NLL from $0.1016$ to $0.0606$, and reduced $\lVert\theta\rVert_2$ from $29.83$ to $14.92$. |
| Dense final classifier | Apply grouped ARD only to hidden transforms; keep the $100$-$10$ classifier dense. | Selected result: $98.05\%$ best full-KL, $97.88\%$ final. |
| Short checkpoint dynamics run | Save every epoch to inspect how importance evolves. | $97.91\%$ best full-KL, $97.67\%$ final; useful diagnostically but inferior to the selected run. |
| $300$-$300$ architecture | Increase the second hidden width. | Dense baseline reached $98.44\%$, but the ARD experiment was abandoned in favor of $300$-$100$. |
| Support-times-SNR importance | Multiply slab-conditioned row SNR by effective slab support. | Rejected: manufactured an artificial floor near $10^{-2}$. |
| Gate-Fisher diagnostic | Estimate functional sensitivity using activation gradients. | Useful as an auxiliary diagnostic, but removed to keep the project weight-posterior based. |
| Group horseshoe | Replace point scales with unregularized continuous horseshoe endpoint scales. | $97.41\%$ best full-KL; strongly shrank layer 1 but left almost all layer-2 neurons active. Rejected. |

The active implementation and formulas are summarized in
[current_method.md](current_method.md).
