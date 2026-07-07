# Security Policy

## Overview

The Butterfly Effect is a **local, offline art-generation tool**. It runs on your
machine to render images from a language model's internals and ships small precomputed
caches so it works with no network access. It exposes **no network service**, accepts no
remote input, and handles **no sensitive or personal data**. The only optional network
activity is a one-time download of the public GPT-2 model via `transformers` when you
ask a model-backed concept to recompute a cache with new parameters.

## Supported Versions

Only the latest `main` branch is supported.

| Version        | Supported          |
| -------------- | ------------------ |
| latest `main`  | :white_check_mark: |
| older commits  | :x:                |

## Reporting a Vulnerability

If you discover a security concern, please report it **privately**:

* Preferred: open a [GitHub Security Advisory](https://github.com/gcgarriga/the-butterfly-effect/security/advisories/new)
  for this repository.
* Alternatively: open a minimal public issue that states a security concern exists
  **without** disclosing details, and request private follow-up from the maintainer.

Please do not disclose the details publicly until a fix is available. We will
acknowledge your report and work with you on a resolution.
