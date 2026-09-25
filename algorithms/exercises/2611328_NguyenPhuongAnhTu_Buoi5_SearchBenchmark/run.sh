#!/usr/bin/env bash
set -euo pipefail
g++ -std=c++17 -O2 -Wall -Wextra src/search_benchmark.cpp -o search_benchmark
./search_benchmark
