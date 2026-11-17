# FlowRouter: stochastic topological lattice surgery
FlowRouter's heuristical objective is to populate a 3d space with lots of potential multi-qubit entangling gates and to select the useful ones for the program. It does this via the following, described in the associated [paper](https://arxiv.org/pdf/2609.23756)
## Pipeline
- Converts the circuit into its ZX-diagram.
- Uses ZX rules to partition the circuit into intra-commuting layers.
- Repeatedly uses max-flow and sequentially more specific choices of sources/sinks to build block graph implementations for gates on the "front" layer.
- Simulates cultivation (given in-program MSC), required delays, and runtime-routed T-gates.
Steps 1-3 are completed with `nomsc.py` and step 4 is completed with `msc.py`. To run all the steps, simply run ```src/nomsc.py``` and then ```src/msc.py``` with the appropriate
## Command-line args
nomsc.py:
- Input circuit via ```--circuit CIRCUIT```.
- Choose in- (edge-of) program MSC with ```--variant {basic,free_magic}```, resp.
- Output block graph HTML with ```--html```.
msc.py:
- Choose distance simulation with ```--distance {11,13,15}```.
## Misc.
The `/augment` folder is used to calculate/explain various rate sampling parameters for MSC, since straight MC simulation is currently unsupported by this program and PureMagic (the main comparison), and slows down the `msc.py` simulation considerably. The code also features an optional Rust accelerator.