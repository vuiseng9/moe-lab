#!/usr/bin/env bash
gid=3

make b4_moedl_e64_k8 gpulist=${gid}
make c1_moedl_s0_k4_e32 gpulist=${gid}