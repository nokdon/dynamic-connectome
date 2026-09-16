# Method

## Task and communication constraint

The bundled problem fixes 24 two-dimensional node positions and a directed candidate mask with 132 edges. Source–target pairs are sampled uniformly from the stored non-self pairs reachable within three candidate-graph hops. Each episode draws an independent payload \(y\sim\mathcal N(0,I_8)\). Node inputs concatenate the source-masked payload, source flag, target flag, and coordinates:

\[
x_i=[1_{i=s}y,\;1_{i=s},\;1_{i=t},\;q_i].
\]

Prediction is made only from the target representation. Loss is MSE averaged over payload coordinates and examples. Global summation of the first eight input channels returns the payload exactly; it is included as an unrestricted control. The task tests constrained communication, not identification of a hidden topology.

## Stage 1: learn a global discrete graph

The router scores candidate destinations and has separate row-open and second-edge gates. In the released configuration, 22 valid rows open, each selecting at most two distinct destinations. Twenty-two open rows do not automatically imply 44 edges: the second gate may close, and candidate availability also matters. All five bundled learned graphs happen to have 44 edges.

The forward graph is binary. Gradients use

\[
Z_{\mathrm{STE}}=Z_{\mathrm{soft}}+\mathrm{stopgrad}(Z_{\mathrm{hard}}-Z_{\mathrm{soft}}).
\]

The first and second destinations are selected without replacement. The backward path retains the original soft gate composition: the second-edge soft mass includes two factors of the open sigmoid, and its hard gate also depends on the thresholded open logit before the exact row-budget mask is applied. This is a historical modeling choice, not a mathematically exact gradient of the hard budget. The implementation preserves it so the published code describes the actual project.

The objective combines payload MSE, a soft noisy-OR path-connectivity loss, entropy, and open/extra-edge penalties. Alternative unnormalized path-mass and combined objectives remain available in the kernel. The default experiment uses noisy-OR. Structural checkpoint selection maximizes admissible-pair reachability minus 0.1 times graph density, using the fixed graph and pair set. It does not use test payloads. Ties keep the earliest candidate.

## Stage 2: freeze structure

For nonempty outgoing rows, \(P_{ij}=Z_{ij}/(\sum_j Z_{ij}+\epsilon)\); empty rows remain zero. Here \(i\to j\) is the edge direction, so incoming aggregation uses **\(P^\top\)**:

\[
H^0=XW_{\mathrm{enc}}+b_{\mathrm{enc}},\qquad
H^{\ell+1}=\operatorname{GELU}(P^\top H^\ell W_{\mathrm{msg}}^\ell
+H^\ell W_{\mathrm{self}}^\ell+b^\ell).
\]

The target representation after three layers is decoded linearly. Encoder/message/readout weights start from the selected Stage 1 state. Stage 2 optimizes these weights with the graph held fixed. The new trainer selects the exact post-update state with the lowest mean MSE on fixed, separately generated validation draws. It includes the initialization as a candidate and retains the earliest state in a tie. L2 regularization is part of training, not the reported MSE.

## B3: hybrid membrane/spike continuation

Stage 2 weights initialize a 12-step recurrent system. Positive/negative payload channels give a signed code. At step \(t\), spikes are computed from the **current** membrane \(U_t\), then messages and \(U_{t+1}\) are computed. The saved spike sequence therefore corresponds to the pre-update membranes. Its binary forward threshold has a sigmoid-derivative surrogate in backpropagation.

The recurrence includes membrane decay/reset, spike messages, recurrent self mixing, decaying input injection, a synaptic trace, and a scaled analog message from \(\tanh U_t\). Readout combines learned weights over the last six target membranes, a target spike-count projection, and a residual GELU MLP before linear decoding. These analog and readout components are why the model is called **hybrid spiking**, rather than an event-only implementation.

Training combines task loss, spike activity regularization, late readout losses, and output/hidden distillation from frozen Stage 2. Teacher outputs are explicit training arguments. The selected historical schedule decays hidden distillation to zero and output distillation to a coefficient floor of 0.15; the latter multiplies the configured output-loss weight. Teacher-free inference is supported. Fully teacher-free training is not a claim about this B3 checkpoint.

## Controls

- **Fixed dense:** uses all candidate edges and trains the same three-layer computation.
- **Random sparse:** samples each row without replacement to match the reference learned graph's outgoing degree. Historical controls use canonical seed 0 as this template.
- **Shortest-path sparse:** counts edges on Euclidean-weighted shortest routes for admissible pairs, ranks rows/edges by route use, and selects within the same 22-row/two-edge budget. The task's graph and pair set are available to this heuristic.
- **Dense soft:** jointly trains masked sigmoid edge weights and the three-layer computation, without hard extraction.

The new baseline trainer uses one shared initialization, common training draws across all four controls, the same validation/test stream convention as the core trainer, and Stage 1 + Stage 2's combined number of update steps. Equal update counts do not establish equal FLOPs or latency. When matching a newly trained graph, supply its artifact directory explicitly.
