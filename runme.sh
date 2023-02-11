#!/bin/sh

set -ex

for i in `seq 1 100`
do
    echo $i
    cargo run --release
done
