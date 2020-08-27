# Invariant Gaussian Processes
This is an implementation of the paper *Learning Invariances using the Marginal Likelihood* by Mark van der Wilk, Matthias Bauer, ST John, and James Hensman [1].

It requires GPflow 2.0.

## Tests
Run the tests using `pytest --cov-report html --cov=invgp`.

- test_sample_SVGP_regression: trains SVGP model, copies parameters to sample_SVGP and matheron_SVGP and checks whether ELBO's are the same
- test_matheron_SVGP_training: train matheron sample_SVGP model from scratch

[1] https://papers.nips.cc/paper/8199-learning-invariances-using-the-marginal-likelihood

## Code style
To format code in a nice and standardised way, simply run:
```
black -t py38 -l 120 filename.py
```
