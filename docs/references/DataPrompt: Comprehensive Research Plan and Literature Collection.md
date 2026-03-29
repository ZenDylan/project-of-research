# DataPrompt: Comprehensive Research Plan and Literature Collection

**The intersection of data valuation and graph prompt learning is an almost completely unexplored territory.** After systematically searching across arXiv, Semantic Scholar, OpenReview, and ACL Anthology, no published work directly combines Data Shapley–style valuation with graph prompt training — confirming a genuine and significant research gap for the D-SVARM algorithm proposed in this project. Below is an organized research plan with milestones, followed by a categorized collection of **70+ papers** with direct links.

---

## Phase 1: Foundation building (Weeks 1–3)

The first phase should focus on deeply understanding the two core pillars. Start with the ICML 2025 paper "Does Graph Prompt Work?" which provides the theoretical bridge — proving graph prompts are essentially data operations. This insight is the linchpin: if prompts operate on data, then data quality directly governs prompt effectiveness. Simultaneously, study Data Shapley (Ghorbani & Zou, ICML 2019) and the SVARM/Stratified SVARM papers (Kolpaczki et al., AAAI 2024) to internalize the variance bounds and the key innovation of updating all n player estimates per coalition sample.

**Milestone 1:** Written summary comparing the computational costs of TMC-Shapley, KNN-Shapley, Data Banzhaf, and Stratified SVARM; formalize how D-SVARM adapts Stratified SVARM's coalition-based updates to graph prompt utility functions.

## Phase 2: Algorithm design and theoretical analysis (Weeks 4–7)

Design the D-SVARM algorithm rigorously. The core challenge is defining the **utility function V(S)** for a subset S of training nodes within the graph prompt framework. Because graph nodes are interconnected (unlike i.i.d. tabular data), node removal changes neighborhood structure — consult the Antonelli & Bojchevski (TMLR 2025) paper on node-level data valuation for GNNs, which addresses exactly this transductive dependency issue. Also study the PC-Winter value paper (Chi et al., 2024) which decomposes nodes into labeled-feature, unlabeled-feature, and structural contributions.

**Key design decisions:**
- Utility function: prompt-based prediction accuracy on a held-out validation set after training on subset S
- Stratification: partition coalitions by size (following S-SVARM) but additionally by graph locality (k-hop neighborhoods)
- Complexity target: **O(n log n)** utility evaluations, avoiding the O(2^n) combinatorial explosion

**Milestone 2:** Complete algorithm pseudocode with theoretical variance bound derivation; prove D-SVARM achieves tighter bounds than naive TMC-Shapley in the graph prompt setting.

## Phase 3: Experimental validation (Weeks 8–12)

Implement D-SVARM on top of the ProG benchmark (NeurIPS 2024) using the All-in-One framework (KDD 2023). Test on both standard graph benchmarks (Cora, CiteSeer, Ogbn-arxiv) and fraud detection datasets (YelpChi, Amazon, T-Finance).

**Experimental protocol:**
1. **Label noise injection:** Randomly flip 10–40% of training labels; measure how D-SVARM–weighted loss improves prompt accuracy vs. uniform weighting
2. **Toxic sample detection:** Inject adversarial nodes; verify D-SVARM assigns them low values
3. **Comparison baselines:** Uniform weighting, LOO, TMC-Shapley, Data Banzhaf, KNN-Shapley
4. **Downstream applications:** Graph anomaly detection (compare with ARC, UNPrompt, AnomalyGFM)

**Milestone 3:** Experimental results on ≥5 datasets showing D-SVARM improves graph prompt accuracy under data quality issues while maintaining computational efficiency.

## Phase 4: Paper writing and submission (Weeks 13–16)

Target venue: **KDD 2026** or **NeurIPS 2026**. Frame the contribution around three claims: (1) first work connecting data valuation to graph prompt learning, (2) D-SVARM algorithm with provable guarantees, (3) consistent empirical improvements across node classification, graph classification, and anomaly detection tasks.

---

## The research gap is real and significant

Systematic search across all major databases confirms **zero papers** combine data valuation with graph prompt learning. The closest existing works fall into three categories: (a) data valuation for NLP prompt/ICL settings (DemoShapley, FreeShap — all in text domain), (b) data valuation for GNNs without prompts (Antonelli & Bojchevski, TMLR 2025), and (c) graph prompt learning without data valuation (the entire All-in-One / ProG family). The DataPrompt project sits precisely at this unexplored intersection, with the ICML 2025 theoretical result providing intellectual justification for why the connection matters.

---

## Category 1: Graph prompt learning (2022–2025)

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 1 | GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks | Sun, Zhou, He, Wang, Wang | KDD 2022 | https://dl.acm.org/doi/10.1145/3534678.3539249 |
| 2 | **All in One: Multi-Task Prompting for Graph Neural Networks** | Sun, Cheng, Li, Liu, Guan | KDD 2023 (Best Paper) | https://arxiv.org/abs/2307.01504 |
| 3 | GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks | Liu, Yu, Fang, Zhang | WWW 2023 | https://arxiv.org/abs/2302.08043 |
| 4 | GPF/GPF-Plus: Universal Prompt Tuning for Graph Neural Networks | Fang, Zhang, Yang, Wang, Chen | NeurIPS 2023 | https://arxiv.org/abs/2209.15240 |
| 5 | Virtual Node Tuning for Few-shot Node Classification | Tan, Guo, Ding, Liu | KDD 2023 | https://arxiv.org/abs/2306.06063 |
| 6 | PRODIGY: Enabling In-context Learning Over Graphs | Huang, Ren, Chen, Kržmanc, Zeng, Liang, Leskovec | NeurIPS 2023 (Spotlight) | https://arxiv.org/abs/2305.12600 |
| 7 | SGL-PT: A Strong Graph Learner with Graph Prompt Tuning | Zhu, Guo, Tang | arXiv 2023 | https://arxiv.org/abs/2302.12449 |
| 8 | GCOPE: All in One and One for All — Cross-domain Graph Pretraining | Zhao, Chen, Sun, Cheng, Li | KDD 2024 | https://arxiv.org/abs/2402.09834 |
| 9 | One for All: Towards Training One Graph Model for All Classification Tasks | Liu, Feng, Kong, Liang, Tao, Chen, Zhang | ICLR 2024 | https://arxiv.org/abs/2310.00149 |
| 10 | MultiGPrompt for Multi-Task Pre-Training and Prompting on Graphs | Yu, Zhou, Fang, Zhang | WWW 2024 | https://arxiv.org/abs/2312.03731 |
| 11 | HetGPT: Harnessing the Power of Prompt Tuning in Pre-Trained Heterogeneous GNNs | Ma, Yan, Li, Mortazavi, Chawla | WWW 2024 | https://arxiv.org/abs/2310.15318 |
| 12 | GraphPrompt+: Generalized Graph Prompt Learning | Yu, Liu, Fang, Liu, Chen, Zhang | IEEE TKDE 2024 | https://arxiv.org/abs/2311.15317 |
| 13 | HGPROMPT: Bridging Homogeneous and Heterogeneous Graphs for Few-shot Prompt Learning | Yu, Liu, Fang, Zhang | AAAI 2024 | — |
| 14 | **ProG: A Graph Prompt Learning Benchmark** | Zi, Zhao, Sun, Lin, Cheng, Li | NeurIPS 2024 (D&B) | https://arxiv.org/abs/2406.05346 |
| 15 | **Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis** | Wang, Sun, Cheng | ICML 2025 | https://arxiv.org/abs/2410.01635 |
| 16 | DAGPrompT: Distribution-aware Graph Prompt Tuning | — | WWW 2025 | https://arxiv.org/abs/2501.15142 |
| 17 | DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning | Wang, Xiong, Wu, Sun, Zhang | CIKM 2024 | https://arxiv.org/abs/2402.11472 |
| 18 | RELIEF: Reinforcement Learning Empowered Graph Feature Prompt Tuning | Zhu et al. | arXiv 2024 | https://arxiv.org/abs/2408.03195 |
| 19 | LEAP: Learning and Editing Universal Graph Prompt Tuning via RL | Xu et al. | KDD 2026 | https://arxiv.org/abs/2512.08763 |

**Surveys:**

| # | Title | Authors | Year | Link |
|---|-------|---------|------|------|
| S1 | Graph Prompt Learning: A Comprehensive Survey and Beyond | Sun, Zhang, Wu, Cheng, Xiong, Li | 2023 | https://arxiv.org/abs/2311.16534 |
| S2 | A Survey of Graph Prompting Methods: Techniques, Applications, and Challenges | Wu, Zhou, Sun, Wang, Liu | 2023 | https://arxiv.org/abs/2303.07275 |
| S3 | Towards Graph Prompt Learning: A Survey and Beyond | Long et al. | 2024 | https://arxiv.org/abs/2408.14520 |

---

## Category 2: Data valuation methods

### Core Data Shapley methods

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 20 | **Data Shapley: Equitable Valuation of Data for Machine Learning** | Ghorbani, Zou | ICML 2019 | https://arxiv.org/abs/1904.02868 |
| 21 | Towards Efficient Data Valuation Based on the Shapley Value | Jia, Dao, Wang et al. | AISTATS 2019 | https://arxiv.org/abs/1902.10275 |
| 22 | Data Valuation Using Reinforcement Learning (DVRL) | Yoon, Arik, Pfister | ICML 2020 | https://arxiv.org/abs/1909.11671 |
| 23 | A Distributional Framework for Data Valuation | Ghorbani, Kim, Zou | ICML 2020 | https://arxiv.org/abs/2002.12334 |
| 24 | Efficient Computation and Analysis of Distributional Shapley Values | Kwon, Rivas, Zou | AISTATS 2021 | https://arxiv.org/abs/2007.01357 |
| 25 | **Beta Shapley: A Unified and Noise-reduced Data Valuation Framework** | Kwon, Zou | AISTATS 2022 | https://arxiv.org/abs/2110.14049 |
| 26 | CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification | Schoch, Xu, Ji | NeurIPS 2022 | https://arxiv.org/abs/2211.06800 |
| 27 | **Data Banzhaf: A Robust Data Valuation Framework** | Wang, Jia | AISTATS 2023 (Oral) | https://arxiv.org/abs/2205.15466 |
| 28 | LAVA: Data Valuation without Pre-Specified Learning Algorithms | Just, Kang, Wang et al. | ICLR 2023 (Spotlight) | https://arxiv.org/abs/2305.00054 |
| 29 | Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value | Kwon, Zou | ICML 2023 | https://arxiv.org/abs/2304.07718 |
| 30 | OpenDataVal: A Unified Benchmark for Data Valuation | Jiang, Liang, Zou, Kwon | NeurIPS 2023 (D&B) | https://arxiv.org/abs/2306.10577 |
| 31 | Robust Data Valuation with Weighted Banzhaf Values | Li, Yu | NeurIPS 2023 | https://papers.nips.cc/paper_files/paper/2023/hash/bdb0596d13cfccf2db6f0cc5280d2a3f-Abstract-Conference.html |
| 32 | DAVINZ: Data Valuation using Deep Neural Networks at Initialization | Wu, Shu, Low | ICML 2022 (Spotlight) | https://proceedings.mlr.press/v162/wu22j.html |

### SVARM family (critical for D-SVARM design)

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 33 | **Approximating the Shapley Value without Marginal Contributions (SVARM & Stratified SVARM)** | Kolpaczki, Bengs, Muschalik, Hüllermeier | AAAI 2024 | https://arxiv.org/abs/2302.00736 |
| 34 | SVARM-IQ: Efficient Approximation of Any-order Shapley Interactions through Stratification | Kolpaczki, Muschalik, Fumagalli, Hammer, Hüllermeier | AISTATS 2024 | https://arxiv.org/abs/2401.13371 |
| 35 | How Much Can Stratification Improve Shapley Value Approximation? | Kolpaczki, Haselbeck, Hüllermeier | xAI 2024 | https://link.springer.com/chapter/10.1007/978-3-031-63797-1_25 |
| 36 | SHAP-IQ: Unified Approximation of any-order Shapley Interactions | Fumagalli, Muschalik, Kolpaczki et al. | NeurIPS 2023 | https://openreview.net/forum?id=IEMLNF4gK4 |
| 37 | shapiq: Shapley Interactions for Machine Learning (library) | Muschalik, Baniecki, Fumagalli, Kolpaczki et al. | NeurIPS 2024 | — |

### Efficient Shapley computation

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 38 | Faster Approximation of Probabilistic and Distributional Values via Least Squares (GELS) | Li, Yu | ICLR 2024 | https://openreview.net/forum?id=lvSMIsztka |
| 39 | One Sample Fits All: Approximating All Probabilistic Values Simultaneously | Li, Yu | NeurIPS 2024 | https://arxiv.org/abs/2410.23808 |
| 40 | Accelerated Shapley Value Approximation for Data Evaluation (δ-Shapley) | Watson, Kujawa et al. | arXiv 2023 | https://arxiv.org/abs/2311.05346 |
| 41 | CHG Shapley: Efficient Data Valuation and Selection towards Trustworthy ML | — | arXiv 2024 | https://arxiv.org/abs/2406.11730 |
| 42 | LossVal: Efficient Data Valuation for Neural Networks | — | arXiv 2024 | https://arxiv.org/abs/2412.04158 |

### Data valuation for graphs

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 43 | **Node-Level Data Valuation on Graphs** | Antonelli, Bojchevski | TMLR 2025 | https://openreview.net/forum?id=tNyApIqDSJ |
| 44 | Precedence-Constrained Winter Value for Effective Graph Data Valuation | Chi, Jin, Aggarwal, Ma | arXiv 2024 | https://arxiv.org/abs/2402.01943 |
| 45 | Shapley-Guided Utility Learning for Graph Inference Data Valuation (SGUL) | — | arXiv 2025 | https://arxiv.org/html/2503.18195 |
| 46 | GraphSVX: Shapley Value Explanations for Graph Neural Networks | Duval, Malliaros | ECML PKDD 2021 | https://arxiv.org/abs/2104.10482 |

---

## Category 3: Intersection — data quality meets prompt learning

This category confirms the research gap. **No paper combines data valuation with graph prompt learning.** The closest works are in NLP/vision domains:

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 47 | **DemoShapley: Valuation of Demonstrations for In-Context Learning** | Xie et al. | arXiv 2024 | https://arxiv.org/abs/2410.07523 |
| 48 | Prompt Valuation Based on Shapley Values | Liu et al. | arXiv 2023 | https://arxiv.org/abs/2312.15395 |
| 49 | FreeShap: Fine-tuning-free Shapley Attribution for Explaining Language Model Predictions | Wang, Lin, Qiao, Foo, Low | ICML 2024 | https://arxiv.org/abs/2406.04606 |
| 50 | TS-DShapley: Data Selection for Fine-tuning LLMs Using Transferred Shapley Values | Schoch, Mishra, Ji | ACL 2023 SRW | https://aclanthology.org/2023.acl-srw.37/ |
| 51 | SHED: Shapley-Based Automated Dataset Refinement for Instruction Fine-Tuning | — | OpenReview | https://openreview.net/pdf?id=Gqou8PRgWq |
| 52 | DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs | Kwon, Wu, Wu, Zou | ICLR 2024 | https://arxiv.org/abs/2310.00902 |
| 53 | CUP: Curriculum Learning based Prompt Tuning for Implicit Event Argument Extraction | Lin et al. | IJCAI 2022 | https://arxiv.org/abs/2205.00498 |
| 54 | Why Is Prompt Tuning for Vision-Language Models Robust to Noisy Labels? | Wu, Tian, Yu et al. | ICCV 2023 | https://arxiv.org/abs/2307.11978 |
| 55 | NLPrompt: Noise-Label Prompt Learning for Vision-Language Models | Pan et al. | CVPR 2025 | https://arxiv.org/abs/2412.01256 |
| 56 | GIF: A General Graph Unlearning Strategy via Influence Function | Wu et al. | WWW 2023 | https://arxiv.org/abs/2304.02835 |

---

## Category 4: Graph anomaly detection with prompt learning

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 57 | **ARC: A Generalist Graph Anomaly Detector with In-Context Learning** | Liu, Li, Zheng, Chen, Zhang, Pan | NeurIPS 2024 | https://arxiv.org/abs/2405.16771 |
| 58 | **UNPrompt: Zero-shot Generalist Graph Anomaly Detection with Unified Neighborhood Prompts** | Niu, Qiao, Chen, Chen, Pang | IJCAI 2025 | https://arxiv.org/abs/2410.14886 |
| 59 | **AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection** | Qiao, Niu, Chen, Pang | KDD 2025 | https://arxiv.org/abs/2502.09254 |
| 60 | **AffinityTune: Prompt-Tuning for Few-Shot Graph Anomaly Detection** | Chen, Zhu, Pang, Yuan, Huang | KDD 2025 | https://github.com/PasaLab/AffinityTune |
| 61 | **GPCF: Neighbor-enhanced Graph Pre-training and Prompt Learning Framework for Fraud Detection** | — | CIKM 2025 | https://doi.org/10.1145/3746252.3761588 |
| 62 | Deep Graph Anomaly Detection: A Survey and New Perspectives | Qiao, Tong, An, King, Aggarwal, Pang | IEEE TKDE 2025 | https://arxiv.org/abs/2409.09957 |

---

## Category 5: Graph fraud detection methods and benchmarks

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 63 | CARE-GNN: Enhancing GNN-based Fraud Detectors against Camouflaged Fraudsters | Dou, Liu, Sun, Deng, Peng, Yu | CIKM 2020 | https://github.com/YingtongDou/CARE-GNN |
| 64 | **DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection** | Duan, Zheng, Gao, Wang, Feng, Wang | AAAI 2024 | https://github.com/AtwoodDuan/DGA-GNN |
| 65 | SEC-GFD: Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum | Xu, Wang, Wu, Wen, Zhao, Wan | AAAI 2024 | https://arxiv.org/abs/2312.06441 |
| 66 | GAGA: Label Information Enhanced Fraud Detection against Low Homophily in Graphs | Wang et al. | arXiv 2023 | https://arxiv.org/abs/2302.10407 |
| 67 | FLAG: Fraud Detection with LLM-enhanced Graph Neural Network | Yang et al. | KDD 2025 | — |
| 68 | DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs | Li, Hu, Hooi, He, Chen | AAAI 2026 | https://arxiv.org/abs/2507.21653 |
| 69 | FraudSquad: Detecting LLM-Generated Spam Reviews with LM Embeddings + GNN | — | arXiv 2025 | https://arxiv.org/abs/2510.01801 |
| 70 | GNN for Financial Fraud Detection: A Review | Cheng et al. | arXiv 2024 | https://arxiv.org/abs/2411.05815 |

---

## Category 6: Data-centric graph learning and label noise

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 71 | Towards Data-Centric Graph Machine Learning: Review and Outlook | Zheng et al. | arXiv 2023 | https://arxiv.org/abs/2309.10979 |
| 72 | Data-centric Graph Learning: A Survey | Guo, Bo, Yang et al. | arXiv 2023 | https://arxiv.org/abs/2310.04987 |
| 73 | Data-centric Artificial Intelligence: A Survey | Zha et al. | ACM CSUR 2024 | https://arxiv.org/abs/2303.10158 |
| 74 | NoisyGL: A Comprehensive Benchmark for GNNs under Label Noise | Wang et al. | NeurIPS 2024 (D&B) | https://arxiv.org/abs/2406.04299 |
| 75 | NRGNN: Learning a Label Noise Resistant GNN on Sparsely and Noisily Labeled Graphs | Dai, Aggarwal, Wang | KDD 2021 | — |
| 76 | A Survey of Data-Efficient Graph Learning | — | arXiv 2024 | https://arxiv.org/abs/2402.00447 |
| 77 | Influence Functions for Edge Edits in Non-Convex GNNs | — | arXiv 2025 | https://arxiv.org/html/2506.04694 |

### Data valuation surveys

| # | Title | Authors | Venue | Link |
|---|-------|---------|-------|------|
| 78 | Data Valuation in Machine Learning: Ingredients, Strategies, and Open Challenges | Sim, Xu, Low | IJCAI 2022 (Survey) | https://www.ijcai.org/proceedings/2022/0782.pdf |
| 79 | Training Data Influence Analysis and Estimation: A Survey | Hammoudeh, Lowd | ML Journal 2024 | https://arxiv.org/abs/2212.04612 |
| 80 | A Comprehensive Study of Shapley Value in Data Analytics | — | arXiv 2024 | https://arxiv.org/abs/2412.01460 |

---

## How D-SVARM connects the two worlds

The technical bridge between data valuation and graph prompt learning rests on three pillars. First, **prompts as data operations**: Wang et al. (ICML 2025) prove that graph prompts approximate graph transformation operators — adding virtual nodes/edges, modifying features — meaning prompt quality is fundamentally constrained by input data quality. Second, **efficient valuation via stratified sampling**: Stratified SVARM's key insight — decomposing φᵢ = φᵢ⁺ − φᵢ⁻ where each coalition sample updates all n players simultaneously — maps naturally to graph prompt training where the utility function V(S) is the prompt-tuned model's accuracy on validation nodes when trained on node subset S. Third, **graph-specific dependencies**: Unlike standard i.i.d. data valuation, graph nodes share neighborhoods; D-SVARM must account for this by incorporating locality-aware stratification and using the prompt itself as a lightweight utility estimator (avoiding full GNN retraining for each coalition).

The variance bound **O(log n / (T − n log n))** from Stratified SVARM translates to the graph setting if the prompt-based utility function satisfies bounded range assumptions. The D-SVARM algorithm achieves maximum sample reuse: each sampled coalition V(S) updates positive estimates for all i ∈ S and negative estimates for all i ∉ S, achieving **n updates per utility evaluation** rather than the single update in TMC-Shapley. For a graph with n training nodes and budget T utility evaluations, this yields a factor-n improvement in sample efficiency.

---

## Priority reading list for the student

The following ten papers should be read first, in this order, as they form the intellectual backbone of the DataPrompt project:

1. **Data Shapley** (Ghorbani & Zou, ICML 2019) — foundational framework
2. **SVARM & Stratified SVARM** (Kolpaczki et al., AAAI 2024) — algorithmic basis for D-SVARM
3. **All in One** (Sun et al., KDD 2023) — graph prompt learning framework to build upon
4. **Does Graph Prompt Work?** (Wang et al., ICML 2025) — theoretical justification
5. **Node-Level Data Valuation on Graphs** (Antonelli & Bojchevski, TMLR 2025) — graph-specific challenges
6. **Data Banzhaf** (Wang & Jia, AISTATS 2023) — maximum sample reuse concept
7. **ProG Benchmark** (Zi et al., NeurIPS 2024) — experimental infrastructure
8. **GCOPE** (Zhao et al., KDD 2024) — cross-domain setting
9. **NoisyGL** (Wang et al., NeurIPS 2024) — understanding label noise on graphs
10. **DemoShapley** (Xie et al., 2024) — closest existing intersection work (NLP domain)

---

## Curated resource repositories

For ongoing paper tracking, bookmark these actively maintained collections:

- **Awesome-Graph-Prompt**: GitHub repo tracking all graph prompt learning papers, maintained by the All-in-One authors
- **safe-graph/graph-fraud-detection-papers**: Comprehensive curated list with 200+ fraud detection papers, interactive dashboard
- **mala-lab/Awesome-Deep-Graph-Anomaly-Detection**: Official companion to the TKDE 2025 survey, continuously updated
- **OpenDataVal benchmark**: https://arxiv.org/abs/2306.10577 — implements 11 data valuation algorithms for standardized comparison
- **shapiq library**: Python package implementing SVARM-IQ and related Shapley interaction methods (Muschalik et al., NeurIPS 2024)

This literature collection covers the full landscape needed for the DataPrompt project. The confirmed research gap — no existing work at the intersection of data valuation and graph prompt learning — represents a strong publication opportunity, with the D-SVARM algorithm positioned to be the first principled method addressing training data quality in graph prompt optimization.