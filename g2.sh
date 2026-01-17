#!/usr/bin/env bash
gid=2

make b3_moedl_e32_k4 gpulist=${gid}
make c2_moedl_s1_k3_e31 gpulist=${gid}