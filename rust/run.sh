#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cargo build --release
exec ./target/release/color_preprocessor "$@"
