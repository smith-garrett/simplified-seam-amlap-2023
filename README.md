# Simplified SEAM, AMLaP 2023 version

**Authors:** Garrett Smith, Maximilian Rabe, Shravan Vasishth, & Ralf Engbert

This is an implementation of Simplified SEAM in Julia that combines the [Simplified SWIFT
model](https://psyarxiv.com/dsvmt/) of eye-movement control with a simple dependency parser.
The model is implemented in Julia (tested with version 1.9) using Pluto notebooks. The
source code for the model is located in the `src` directory. The Schilling corpus and the
parses are located in the `data` directory. The full SEAM model can be found [here
(paper)](https://arxiv.org/abs/2303.05221) and [here (code)](https://osf.io/ad5nx/).

The parameter fitting was done using the [DIME](https://github.com/gboehl/DIMESampler.jl)
algorithm. This is done the `fitting-dime.jl` Pluto notebook. This is also where the model
comparison code is. The `posterior-plots-amlap.jl` generates some summary plots of the
posterior distributions of model parameters.

This code is a work in progress, but it should be possible to replicate the results we
reported at AMLaP 2023 in San Sebastián. Please get in contact if you run into problems!

