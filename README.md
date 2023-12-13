# ScoreMatchingVI.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/ScoreMatchingVI.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/ScoreMatchingVI.jl/dev/)
[![Build Status](https://github.com/ptiede/ScoreMatchingVI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/ScoreMatchingVI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/ScoreMatchingVI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/ScoreMatchingVI.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

This is an implementation of [Variational Inference with Gaussian Score Matching](https://arxiv.org/abs/2307.07849) by Modi et al. (2023) that attempts to find a Gaussian approximation of an arbitrary distribution by matching their score function (gradient of the log-density) are equal. By making a Gaussian approximation, the paper developed a closed-form solution to the optimization problem, making the implementation of the algorithm rather trivial (< 100 LOC) and essentially hyperparameter free. 

**This is currently in the alpha stage and has not been optimized or tested fully**
