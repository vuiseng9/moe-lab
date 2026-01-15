#!/usr/bin/env bash
gid=1

make b2_moedl_e16_k2 gpulist=${gid}
make c3_moedl_s2_k2_e30 gpulist=${gid}