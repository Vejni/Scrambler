#!/usr/bin/env bash
# Clone the original implementation (code, example data and Keras predictors) into reference/scrambler.
set -euo pipefail
cd "$(dirname "$0")"
[ -d scrambler ] || git clone --depth 1 https://github.com/johli/scrambler scrambler
