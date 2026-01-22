<img src="assets/compare_lb_strategy_heatmaps.gif" width="600" style="height:auto;">

## MoE Ablations
**Jump to:**
* Load Balancing Strategy: Loss Penalty vs. Router Biasing
* Future Plans

**Motivation:** 

2025 is the year of reasoning and agents. It can also be argued as the year Mixture-of-Experts (MoE) models truly hit the mainstream. Virtually every flagship model from frontier labs is an MoE, although Google has been pioneering and popularizing the idea since [2017][og-moe-2017].

The beginning of 2025 was marked by the release of DeepSeek R1, which demonstrated reasoning learned through pure RL. Its backbone is DeepSeek V3, a 671B MoE. Qwen3 MoE arrived mid-year, followed by OpenAI's GPT-OSS in late August. The year closed with models such as [Kimi-K2][kimi-k2-report] and [Mistral Large 3][ml3-blog], which largely adopt DeepSeek V3–style architectures with an even larger number of experts.

Personally, I am particularly interested in understanding which components of MoE models contribute most to their efficacy beyond *just* scale and the conditional computation. My initial questions centered on how effective the load-balance biasing strategy used in DeepSeek V3 is compared to loss-based penalties. I did not encounter comparative study on this technique. *(I may have overlooked prior ablations and would appreciate pointers)*. I also questioned whether shared experts are truly mandatory, given that they introduce an additional engineering effort. While some prior work suggests they are not strictly required, I find the existing evidence not yet compelling *(though personally favor avoiding unremarkable additions)*. 

I intended to carry out these ablations using modest resources with HuggingFace (HF) Transformers. However, I could not find an implementation that exposes load balance biasing control. More importantly, there was no single MoE model type that allowed meaningful ablations while keeping most components consistent and only contrasting a single axis of design choices. For example, when comparing loss penalties versus biasing strategies for balancing expert load, I would like the attention layers to remain identical. DeepSeek V3 uses Multi-Latent Attention (MLA), which differs from the standard multi-head attention (MHA/MQA/GQA) used in models such as OLMoE or Qwen3, making direct comparisons unhygienic.

As a result, I decided to implement a new model type in local HF Transformers, `Moedl` (no pun intended!😝). This allows individual design choices to be turned on or off in a controlled manner while keeping the rest of the architecture fixed. While some features are still work in progress and certain ablations require larger resources, I believe there is now sufficient material to document the observations and findings.



**Hit the ground running with:**
* **Install**: clone and `make install-moelab` or `make install-dev-moelab`
* **Test**: `make run-tests`
* **Run**: 
    * `make <experiment-id>`, see [Makefile][mkfile]. Most experiments can fit within a single 80GB GPU. Each took about 3 hrs on a RTX Pro 6000 gpu.
        ```bash
        # Available make targets (Experiments)
        00_llama2_ref        b20_moedl_e4_k1_4ep  e1_moedl_cf_1.0
        01_moedl_dense       c1_moedl_e8_k1       e2_moedl_cf_1.5
        a0_moedl_no_lb       c2_moedl_e16_k2      e3_moedl_cf_2.0
        a1_moedl_lb_penalty  c3_moedl_e32_k4      e4_moedl_cf_2.5
        a2_moedl_lb_biasing  c4_moedl_e64_k8      gen-tinystories
        b1_moedl_e2_k1       d1_moedl_s0_k4_e32   gpulist-check-busy
        b2_moedl_e4_k1       d2_moedl_s1_k3_e31   install-dev-moelab
        b3_moedl_e8_k1       d3_moedl_s2_k2_e30   install-moelab
        b4_moedl_e16_k1      d4_moedl_s3_k1_e29   run-tests
        ``` 
    * **More customization**: use `moelab_main.py` like we use standard HF script. Do python `moelab_main.py --help` to see options.
    * **Find LR**: Appending `--sweep_lr <list of comma-limited lr>` to `moelab_main.py` will turn it into learning rate sweep over predefined values for small number of steps which can be configured with `--sweep_lr_steps <num_steps>`. For experiments in the Makefile, just append sweep_lr=1 to the make command. e.g. `make c1_moedl_e8_k1 sweep_lr=1`. A report will be generated in the output folder and metrics of respective sweare logged to wandb.

* All runs are shared on [W&B project][wbproj]. 

* Qualitative Eval per [TinyStories][ts-paper].
    ```bash
    make gen-tinystories ckpt=roneneldan/TinyStories-33M  # official dense GPTNeo model
    make gen-tinystories ckpt=vchua/moelab-e2-k1-4ep-tinystories  # moedl dense model
    make gen-tinystories ckpt=vchua/moelab-e4-k1-4ep-tinystories  # moedl moe model
    ```

---
### Scope and Implementation
Our objectives are:
1. To enable controlled ablations of key MoE design choices.
2. To support small-scale experiments that require only a single GPU and can be completed within a few hours.
3. To use a simple, accessible and familiar stack, entirely within the HuggingFace Transformers ecosystem.

#### [TinyStories][ts-paper]
All experiments use TinyStories, a synthetic [dataset][ts-ds] of short stories restricted to vocabulary typically understood by children of ages 3-4, generated using GPT-3.5 and GPT-4.

We choose TinyStories because the original paper demonstrates that models with as few as ~10M parameters can already learn to generate fluent and logically consistent stories. This allows us to bound both model size and training compute, keeping experiments feasible on a single GPU while still uncovering meaningful ablation trends. MoE models in this repo are typically a few hundred million parameters, mostly around 400M total parameters.

TinyStories is also easy to evaluate qualitatively: story coherence and logical consistency are readily observable, making it practical for comparing MoE ablations at small scale. In contrast, prior experience using GPT-2 or OPT models of similar scale trained on large, generic corpora often results in incoherent or unstructured generation, making qualitative comparison across models unreliable or infeasible.

We use the LLaMA-2 tokenizer for its smaller 32K vocabs. After tokenization, the training set contains approximately XXX tokens. We limit most experiments to 2 epochs, based on the diminishing returns of longer epochs observed in [*Scaling Data-Constrained LMs*][dc-illa].

#### [`Moedl` Configurables][MoedlCfg] & [Implementation][MoedlImpl]

This repo primarily focuses on the MoE layers. We need a smaller vocab given the property of TinyStories. Hence, we inherit from the `Llama` (LLaMA-2) architecture as the starting point, 32K vocab. Its attention stack also reflects the current standard: configurable GQA (MHA), RoPE, RMSNorm, and GLU-style MLPs. As a result, the standard LLaMA configurables are directly applicable to `Moedl`, including `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_hidden_layers`, and related parameters.

`Moedl` can be configured as either a dense or an MoE model. As a sanity check, we train an equivalent LLaMA-2 dense model (`make 00_llama2_ref`) and a `Moedl` dense model (`make 01_moedl_dense`) on TinyStories. Equivalent training trajectories ensure architectural parity.

For MoE configurations, the following parameters can be varied:

* `num_experts`: number of experts per MoE layer, denoted as **E** throughout. If set to 0, a dense layer is used.
* `num_active_experts`: number of experts activated per token, denoted as **K** throughout.
* `num_shared_experts`: number of shared experts across all layers, denoted as **ES** throughout.
* `lb_coeff`: load-imbalance penalty coefficient. If set to 0, no loss-based penalty is applied.
* `lb_gamma`: router biasing update rate. If set to 0, bias-based control is disabled.
* No load balancing is applied when both `lb_coeff` and `lb_gamma` are set to 0.
* `capacity_factor`: expert capacity factor controlling token dropping. Setting this to 0 disables token dropping (i.e., dropless routing).

Tests are developed to verify against Olmoe and DeepSeek V3 to ensure MoE implementation correctness.

#### [`MoedlTrainer`][MoedlTrainer]

We subclass the HF `Trainer` to add MoE-specific training bookkeeping and logging, including routing statistics, expert load tracking, and fine-grained expert load heatmap generation with GIF collation. Of particular note is the `LoadBalanceBiasController`, which encapsulates router biasing control for load balancing; this is discussed in more detail in the load-balance strategy section. I encourage interested readers to review the [code][MoedlTrainer], which is self-explanatory with comments.

<!-- tests? -->

---
### Load Balancing Strategy 
> TLDR: Router biasing is surprisingly effective, easier to implement, requires less tuning!

Load balancing is crucial for MoE models to ensure that experts or more importantly the underlying devices are utilized effectively. By *load balance*, it simply means router's ability to distribute the incoming tokens evenly across all experts. Proper load balance allows computational work to be shared evenly across experts/devices, enabling parallelism and improving both training and inference efficiency.

#### Load Imbalance Penalty
Google has pioneered the use of an *auxiliary loss* added to the training objective to penalize load imbalance among experts, encouraging models to learn more even routing in order to achieve a lower overall training loss.

&emsp; $\mathcal{L} = \mathcal{L}_{\text{ce}} + \lambda \, \mathcal{L}_{\text{aux}}$ &emsp; 

&emsp; $\mathcal{L}_{\text{aux}} = E \sum_{e=1}^{E} f_e \, p_e$ &emsp; ── eq.1 

&emsp; where &emsp; 
* $E$ is the number of experts
* $f_e$ is the fraction of tokens routed to expert $e$
* $p_e$ is the average router probability assigned to expert $e$ (router's softmax output, averaged over tokens and $k$ experts)

This technique is conceptually simple and closely resembles regularization. In our implementation, $\lambda$ is controlled by the `lb_coeff`.

#### Router Load Biasing (DeepSeek v3)
Dubbed as *auxiliary loss-free load balancing*, DeepSeek V3 introduced an alternative approach that directly modifies the router logits (sigmoid output) with an additive *load-biasing term* before applying the Top-k assignment of experts. This biasing term is updated periodically based on the observed expert loads, effectively nudging the router to favor less-utilized experts.

&emsp; $\mathbf{s} = \sigma(W x) + \mathbf{b}_{\text{lb}}$ &emsp; ($\sigma$ is sigmoid instead of softmax)

&emsp; $\mathcal{K} = \operatorname{TopK}(\mathbf{s}, K)$ &emsp; (expert selection)

&emsp; where &emsp;
* $\mathbf{s} \in \mathbb{R}^{E}$ is the biased router score vector over all experts (per token).
* $\mathbf{b}_{\text{lb}} \in \mathbb{R}^{E}$ is the load-bias vector, initialized to $\mathbf{0}$ and updated periodically as:

&emsp; $\mathbf{b}_{\text{lb}} \leftarrow \mathbf{b}_{\text{lb}} + \gamma\,(\bar{\mathbf{f}} - \mathbf{f})$ &emsp; (every $N$ steps)

&emsp; where &emsp;
* $\mathbf{f} \in \mathbb{R}^{E}$ is the observed expert-load fraction averaged over $N$ steps.
* $\bar{\mathbf{f}} = \tfrac{1}{E}\mathbf{1}$ is the uniform target load.
* $\gamma$ is the bias update rate.

It is important to note that $b_{lb}$ does not participate in forward propagation other than modifying the router scores, hence does not affect the gradient computation. Its value is calibrated solely through the periodic update based on observed loads. If you are familiar with control systems, this is equivalent to a simple proportional controller operating directly on the routing imbalance error signal.

#### Ablation Results & Analysis
Setup: We ablate on `Moedl` with 8 experts (E=8) and 1 active expert per token (K=1) on TinyStories for 2 epochs We compare **no load balancing** (baseline), **imbalance penalty**, and **router biasing**.

| make [experiment id]         | Eval Loss |
|---------------------------   |:---------:|
| `a0_moedl_no_lb`             | 1.127     |
| `a1_moedl_lb_penalty`        | 1.137     |
| `a2_moedl_lb_biasing`        | 1.130     |

At first glance, all three strategies converge to similar final eval loss, with the best result achieved without load balancing, followed closely by router biasing, and finally the imbalance penalty. While the final loss differences are small, the load-balancing dynamics differ. To illustrate this, we examine expert load over time. The following plots, logged in W&B, show expert load averaged across experts and layers.

**Without load balancing**, expert imbalance is immediately apparent. A small subset of experts dominates the routing, as reflected by the disproportionate heights in the stacked plot and the uneven distributions in the % overlay view.

<img src="assets/a0_no_lb_expert_load.png" width="600" style="height:auto;">

**Imbalance penalty** improves load distribution gradually over training. Between steps 2k and 10k, the % plot exhibit noticeably higher variance, indicating noisy and unstable routing before the auxiliary loss sufficiently regularizes expert utilization.

<img src="assets/a1_lb_penalty_expert_load.png" width="600" style="height:auto;">

**Router biasing** yields the most stable behavior. Expert load converges rapidly toward a uniform distribution, with significantly tighter variance bounds in the % plot throughout training.

<img src="assets/a2_lb_biasing_expert_load.png" width="600" style="height:auto;">

At this point, router biasing edges out the imbalance penalty (lower eval loss, tighter distribution variance), but not convincingly so. The *take-my-money* moment comes next: per-expert, per-layer load deviations from balance point visualized as animated heatmaps over training.

![](assets/compare_lb_strategy_heatmaps.gif)

Examining the animated heatmaps, the advantage of router biasing becomes strikingly clear. Notice how  plain and less hot the color distribution remains throughout training. Router biasing rapidly achieves near-perfect uniform balance across experts and layers, with minimal deviation over time. In contrast, the imbalance penalty shows signs of expert collapse in later layers during the later stages of training, where certain experts remain consistently overloaded while others are underutilized. As expected, the no-load-balancing baseline exhibits imbalance across layers throughout training.

**Why** does router biasing work better? The auxiliary loss in Eq. (1) is a *globally* reduced scalar objective, a few localized imbalance signals may be too weak to meaningfully influence the overall loss. Router biasing, by contrast, applies control directly at a per-router level. Each expert is adjusted independently via a dedicated bias term, enabling more precise and effective correction.

While expert-specific coefficients could be introduced for the imbalance penalty, doing so requires additional tuning. Router biasing is simpler to implement and requires minimal tuning in practice. In my experience, tuning is straight forward, basically ensuring the bias update rate $\gamma$ is not overly large.

Based on these results, we adopt router biasing as the default load-balancing strategy for the remaining ablations.



Acknowledgements

Citations

[mkfile]: ./Makefile
[wbproj]: https://wandb.ai/vchua/moe-lab
[MoedlCfg]: ./src/moelab/moedl/configuration_moedl.py
[MoedlImpl]: ./src/moelab/moedl/modeling_moedl.py
[MoedlTrainer]: ./src/moelab/moedl/trainer.py

[olmoe]: https://arxiv.org/abs/2409.02060
[ec-paper]: https://arxiv.org/abs/2202.09368
[ml3-blog]: https://mistral.ai/news/mistral-3
[kimi-k2-report]: https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
[og-moe-2017]: https://arxiv.org/abs/1701.06538 
[ts-paper]: http://arxiv.org/abs/2305.07759
[ts-ds]: https://huggingface.co/roneneldan/TinyStories-33M

[dc-illa]:http://arxiv.org/abs/2305.16264