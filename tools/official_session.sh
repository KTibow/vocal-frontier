#!/bin/bash
# The official session on the Ryzen mini PC: native, browser CPU, Vega. See MAINTAINING.md. usage: tools/official_session.sh <site url> <model...>
cd "$(dirname "$0")/.."; exec .venv/bin/vf official all "$@"
