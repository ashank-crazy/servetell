#!/usr/bin/env bash
# Download the Match Charting Project files (CC BY-NC-SA 4.0, Jeff Sackmann / Tennis Abstract) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
base="https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
for tour in m w; do
  for f in matches points-to-2009 points-2010s points-2020s; do
    echo "charting-$tour-$f.csv"
    curl -fsSL "$base/charting-$tour-$f.csv" -o "data/raw/charting-$tour-$f.csv"
  done
done
