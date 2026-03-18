#!/usr/bin/env bash
gid=2

make b2_moedl_e4_k1 gpulist=${gid}
make c2_moedl_e16_k2 gpulist=${gid}
make d1_moedl_s0_k4_e32 gpulist=${gid}
make d4_moedl_s3_k1_e29 gpulist=${gid}
make e3_moedl_cf_2.0 gpulist=${gid}