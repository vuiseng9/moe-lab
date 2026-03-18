#!/usr/bin/env bash
gid=3

make b3_moedl_e8_k1 gpulist=${gid}
make c1_moedl_e8_k1 gpulist=${gid}
make d2_moedl_s1_k3_e31 gpulist=${gid}
make d3_moedl_s2_k2_e30 gpulist=${gid}
make e4_moedl_cf_2.5 gpulist=${gid}