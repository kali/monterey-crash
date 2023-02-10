#!/bin/sh

set -ex

for i in `seq 1 100`
do
    echo $i
    RUSTFLAGS=-Awarnings cargo test --release -- --nocapture
done
