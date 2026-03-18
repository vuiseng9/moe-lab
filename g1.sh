#!/usr/bin/env bash
gid=1

make a1_moedl_lb_penalty gpulist=${gid}
make a2_moedl_lb_biasing gpulist=${gid}
make b4_moedl_e16_k1 gpulist=${gid}
make c3_moedl_e32_k4 gpulist=${gid}
make e2_moedl_cf_1.5 gpulist=${gid}