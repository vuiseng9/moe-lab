#!/usr/bin/env bash
gid=0

make a0_moedl_no_lb gpulist=${gid}
make b1_moedl_e2_k1 gpulist=${gid}
make c4_moedl_e64_k8 gpulist=${gid}
make e1_moedl_cf_1.0 gpulist=${gid}