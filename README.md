# cubempc

Layered MPC (CUBE) — protocol code and tests for verifiable secret sharing and secure computation.

## Background

This repository is an experimental implementation of layered MPC protocols based on Shamir secret sharing. The goal is to study how verifiable secret sharing, resharing, random generation, multiplication, and circuit evaluation can be organized across logical committees while tolerating Byzantine faults.

The implementation uses finite-field arithmetic and robust Reed-Solomon style reconstruction. It is intended for protocol experiments and correctness tests, not as a production cryptographic library.

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```
