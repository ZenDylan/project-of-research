
科研选题报告
Research Topic Proposal for ROMA Lab Internship

图提示学习与社交媒体虚假信息检测的交叉研究

申请人：赵博恩
合肥工业大学
指导老师：孙相国 教授
2026年3月
 
致孙老师及ROMA Lab团队
孙老师好！感谢您给予的这次科研实习机会。收到您的邮件后，我这几天主要在做两件事：一是研读ROMA Lab近三年发表的论文，尤其是图提示学习和社交媒体分析两条主线的工作；二是调研图提示学习在异常检测、欺诈检测方向的最新进展，看看有没有值得尝试的交叉点。
结合我自己在数据估值和虚假评论图网络检测方面的一些经验，我整理了五个初步的选题方向。这些想法还比较粗糙，恳请老师和团队评估指正。哪些方向值得深入、哪些需要调整、或者老师有其他建议的方向，我都非常愿意听取。
一、研究背景与调研综述
1.1 对ROMA Lab技术路线的理解
通过阅读实验室的论文，我把ROMA Lab的研究脉络理解为四条相关联的线：
第一条是图提示学习的方法论。从All-in-One（KDD'23 Best Paper）提出多任务统一的图提示框架，到GCOPE（KDD'24）处理跨域预训练中的负迁移，到ProG（NeurIPS'24）做标准化基准，再到"Does Graph Prompt Work?"（ICML'25）从数据操作的角度做理论分析——这条线从方法到工具到理论都比较完整了。特别是ICML'25的工作让我印象深刻，它说明图prompt实质上是一种可学习的数据变换算子，这个结论对后面怎么设计和优化prompt很有启发。
第二条是社交媒体上的虚假信息检测。"Nowhere to Hide"（TNNLS'24）做的是基于转发图的谣言检测，TKDE'22那篇则是用超图建模的动态谣言检测。这两项工作加上实验室名称ROMA本身（Responsible Online Media Analytics），说明虚假信息治理是实验室长期关注的核心问题。
第三条是图稀疏化。MoG（ICLR'25 Spotlight）用MoE的思路给每个节点动态分配稀疏化策略，在ogbn-proteins上去掉30%的边还能保持性能，对做大规模图的高效推理很有用。
第四条是图公平性。FROG（CIKM'25）发现图上删节点/删边会打破同质性、放大歧视，于是设计了重布线策略。这一块和欺诈检测的结合我觉得很有前景，后面的选题五会具体展开。
1.2 图提示学习在异常/欺诈检测中的进展
这部分是我花时间最多的调研。2024-2025年这个交叉方向有不少新工作冒出来：ARC（NeurIPS'24）做了基于in-context learning的通用图异常检测；UNPrompt（IJCAI'25）通过邻域prompt实现了零样本检测；AnomalyGFM（KDD'25）是第一个面向图异常检测的基础模型；AffinityTune（KDD'25）设计了多粒度的prompt tuning。另外GPCF（CIKM'25）把图预训练+prompt落地到了腾讯微信支付的交易欺诈场景。
但我注意到一个空白：这些工作主要在通用图异常检测基准上做验证（社交网络、引文网络等），还没有人把图提示学习系统地应用到虚假评论检测这种特定场景——也就是用户-商品-评论构成的异质图。虚假评论图有它自己的难处，比如类不平衡很极端（通常欺诈<5%）、欺诈者会故意和正常用户建立连接来伪装自己（camouflage），这些特性让通用方法不太能直接搬过来用。我认为这里有明确的研究空间。
1.3 GNN欺诈检测的最新动态
GNN欺诈检测这两年变化很快。一个突出的趋势是大家开始重视异质性处理——因为欺诈者倾向于和正常用户建立连接，导致传统基于同质假设的GNN效果打折。SEC-GFD从频谱域处理这个问题，DGA-GNN（AAAI'24）用动态分组聚合在YelpChi、Amazon等5个基准上超了12个基线。另外LLM和GNN的结合是2025年最明显的趋势，FraudSquad把语言模型嵌入和图Transformer结合后在检测LLM生成的虚假评论上效果提升很大。
还有一个值得关注的现象：零样本的LLM检测器对虚假评论的检测成功率不到45%，说明光靠文本分析不够，图结构信息的引入变得更加关键。
1.4 关于谣言检测与虚假评论检测的统一建模
这一点是我在调研中比较意外的发现——谣言检测和虚假评论检测虽然都是虚假信息问题，但我翻了不少论文，没有找到在图层面把两者统一建模的工作。两者的图形态不同（传播图 vs. 用户-商品-评论图），但共享一些关键信号，比如欺骗性文本模式、用户行为突然爆发、图上的异常聚集。All-in-One的多任务统一思路（通过induced graph把不同级别任务统一到图分类）理论上可以桥接这两种不同形态的图，这是我选题二的出发点。
二、个人研究背景
我之前的研究主要涉及两块：
一是数据估值。我接触过Data Shapley、Beta Shapley这些方法，理解如何量化单个数据点对模型的贡献。这个背景和ICML'25那篇 "Does Graph Prompt Work?" 有自然的衔接——既然图prompt本质上是数据操作，那数据质量对prompt效果的影响就值得研究。
二是基于图网络的虚假评论检测。我用PyTorch和DGL/PyG搭过用户-商品-评论的图，跑过一些检测实验。这部分经验应该能帮我比较快地上手ProG工具箱和ROMA Lab的相关代码。
基于以上调研和个人经历，我把选题方向聚焦在：将ROMA Lab的图提示学习技术应用到虚假信息检测场景，同时在合适的选题中引入数据估值的视角。
三、选题提案
选题一：PromptFraud —— 基于图提示学习的少样本虚假评论检测
问题动机
虚假评论的标注成本很高，真实场景中有确认标签的数据往往不到5%。新平台、新品类上线时更是几乎没有历史标注可用。目前主流的全监督GNN方法（比如DGA-GNN、CARE-GNN）一旦标签减少，性能下降很明显。而All-in-One已经在通用图任务上证明了10-shot设置可以超越全量微调，但据我调研，还没有人把图提示学习搬到虚假评论检测的异质图上来做。我觉得这里有一个比较清晰的gap。
技术路线
大致分三步。先在大规模评论图（比如YelpChi）上做自监督预训练（边预测、GraphCL、GraphMAE等策略），让模型学到用户-商品-评论交互的通用模式。然后设计评论图专用的prompt——在特征层给不同类型节点加可学习向量，在结构层构建可学习子图来捕获典型欺诈行为模式（比如burst review pattern），在任务层通过induced graph把节点级检测转化为图分类。最后用MAML做元学习优化prompt初始化，实现5-shot/10-shot快速适配。
有个技术细节需要特别处理：评论图的异质性很强（欺诈者故意和正常用户连接），所以prompt聚合时需要区分同质边和异质边，施加不同权重。
预期创新点
（1）把图提示学习系统地引入虚假评论检测，填补review graph场景的方法空白；（2）针对评论图的伪装异质性设计专门的prompt结构；（3）在ProG上做多种预训练+prompt组合的系统评估，为后续研究提供baseline。
数据集与实验
YelpChi（67K节点，14.5%欺诈率）、Amazon（11K节点，6.87%欺诈率）、T-Finance（百万级），都集成在DGL的FraudDataset里。实验设计：full-shot/50-shot/10-shot/5-shot对比，基线包括CARE-GNN、DGA-GNN、AnomalyGFM、UniGAD、MetaGAD等，核心指标AUC-ROC和F1-macro。ProG工具箱可以直接复用，开发效率应该比较高。
选题二：UniMisinfo —— 谣言与虚假评论的统一图提示检测框架
问题动机
如前面1.4节提到的，谣言检测和虚假评论检测目前是两个独立的研究社区，我没有找到在图层面把它们统一起来的工作。但ROMA Lab恰好在两边都有积累——All-in-One的多任务统一能力和谣言检测的图建模经验（TNNLS'24, TKDE'22）。如果能用一个共享的框架同时处理两类任务，不仅方法论上有突破，对实验室"Responsible Online Media Analytics"的整体定位也非常匹配。
技术路线
核心思路是把传播图和评论图统一表示为异构信息网络，定义标准化的节点类型（User、Content、Entity）和边类型（Post、Reply、Review、Purchase）。在混合数据上预训练学习通用的欺骗模式，然后分别给两个任务设计task-specific prompt——谣言检测prompt捕获传播的速度、广度、深度，虚假评论prompt捕获行为突发性和评论集中度。可以借鉴GCOPE的Graph Coordinator来处理两种图之间的分布差异。最终通过prompt空间的对齐来实现谣言和虚假评论之间的知识迁移。
预期创新点
（1）据我所知是首个在图层面统一两种虚假信息检测的框架；（2）有望发现并量化两类任务间共享的图模式；（3）实现从一种虚假信息到另一种的跨任务知识迁移。这个方向的创新空间很大，不过实现起来也会比选题一复杂不少，可能需要更多时间打磨。
数据集与实验
谣言：Twitter15/16、PHEME、Weibo。虚假评论：YelpChi、Amazon。实验要做的包括：单任务性能（分别和各自领域的SOTA比）、跨任务迁移（谣言数据训练后zero/few-shot检测虚假评论，反之亦然）、多任务联合训练vs单任务的对比、可迁移模式的可视化分析。
选题三：SparseFraud —— 基于混合图稀疏化的高效欺诈检测
问题动机
大规模欺诈图（比如T-Finance有百万级节点）跑GNN很慢，而且欺诈者伪装产生的噪声边还会拉低检测精度。MoG（ICLR'25 Spotlight）已经证明了MoE式动态稀疏化的效果，但它是在通用图任务上验证的，没有专门考虑欺诈图的特点。欺诈图里有两种很不一样的边——camouflage边（欺诈者和正常用户之间的伪装连接，应该被移除）和collaboration边（欺诈团伙内部的关联，应该被保留），这种区分在MoG原始设计里没有体现。
技术路线
在MoG的MoE框架上加入欺诈检测专用的expert：一个关注同质性（保留同类节点间的边），一个关注拓扑异常（保留揭示异常聚集的边），一个专门检测并降权camouflage边。同时设计一个团伙保护损失，防止稀疏化过程把欺诈者之间的关键关联边也切断了。稀疏化后的图可以直接喂给任何下游GNN检测器，相当于一个即插即用的预处理模块。
预期创新点
（1）把MoE图稀疏化引入欺诈检测，设计异常感知的裁剪准则；（2）为camouflage边识别这个老难题提供新角度；（3）能同时加速推理和提升性能（去掉的是噪声边）。这个选题的好处是MoG有开源代码，改动量相对可控，出实验结果应该比较快。
数据集与实验
YelpChi、Amazon、T-Finance、T-Social、Elliptic。核心实验包括：不同稀疏率（10%-50%）下的AUC-ROC/F1变化曲线、稀疏化前后的训练/推理时间和内存对比、与DGA-GNN/CARE-GNN/GAGA等基线的性能比较。特别想做一个case study，看看被模型移除的边是不是真的以camouflage边为主。
选题四：DataPrompt —— 数据估值引导的图提示学习优化
问题动机
这个选题是从我自己的数据估值背景出发的。ICML'25证明了图prompt是数据操作，那一个自然的问题就是：如果训练数据本身有质量问题（标签噪声、有毒样本），prompt学习会受多大影响？现有方法对所有训练样本一视同仁，没有考虑数据质量的差异。而数据估值方法可以量化每个样本的贡献，但现有的图数据估值方法（Antonelli & Bojchevski, TMLR 2025）计算成本太高，不能直接用在prompt训练里。
技术路线
设计一种轻量级的图数据估值方法——利用prompt本身作为数据操作工具来快速估计不同数据子集的效用，避免完整重训练。在prompt梯度更新时，按估值给每个训练节点的损失加权：高价值的加大权重，噪声样本降权。理论上希望能在ICML'25的误差框架下推导出数据质量对prompt近似误差的影响上界。
预期创新点
（1）把数据估值和图提示学习两个方向连起来；（2）建立数据质量与prompt效果之间的理论关系；（3）设计高效的图数据估值近似算法。这个方向理论性比较强，实现难度也最大。如果选这个方向，我可能需要老师在理论方面给比较多的指导。
选题五：FairPromptFraud —— 面向公平性的欺诈检测图提示框架
问题动机
GNN欺诈检测有一个容易被忽视的问题——对某些用户群体的误判率偏高。FairGAD（Neo et al., CIKM'24）正式定义了这个问题，不过他们测试了9种GAD方法配合5种公平方法，发现现有公平方法在图数据上效果不太理想。另一边，FPrompt（Li et al., WWW'25）刚刚把公平性引入了图prompt tuning，但没涉及异常检测。把FROG（CIKM'25）的公平性理论、FPrompt的prompt设计和欺诈检测结合起来，我觉得可以做出有意义的工作。
技术路线
设计一个双prompt结构：一组效用prompt负责优化检测精度，一组公平prompt通过调整敏感属性相关的信息传播来实现组间公平。利用prompt的轻量化优势做跨平台的公平迁移检测。理论上想基于FROG的dyadic fairness上界推导prompt操作对公平性的影响。
预期创新点
（1）同时优化检测精度和公平性的图prompt欺诈检测框架；（2）把公平图prompt tuning扩展到异常检测场景；（3）建立prompt操作与公平性指标的理论联系。数据集可以用FairGAD提供的Reddit和Twitter数据集，以及YelpChi。
四、后续计划
以上五个选题是我目前能想到的方向，肯定还有很多不成熟的地方。在与老师确定具体方向后，我打算这样推进：
先花一周左右精读所选方向的核心论文（包括ROMA Lab的相关工作），把技术细节吃透。然后用一到两周搭好实验环境，跑通ProG或MoG的代码，熟悉数据集和评估流程。之后就可以和老师讨论确定具体的论文题目和创新点，正式进入研究。
感谢老师和团队抽时间看这份报告。期待后续的交流！

赵博恩
合肥工业大学
2026年3月21日
 
附录：主要参考文献
[1] Sun, Cheng, Li, Liu, Guan. All in One: Multi-task Prompting for Graph Neural Networks. KDD 2023 (Best Paper).
[2] Zhao, Chen, Sun, Cheng, Li. All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining. KDD 2024.
[3] Zi, Zhao, Sun, Lin, Cheng, Li. ProG: A Graph Prompt Learning Benchmark. NeurIPS 2024.
[4] Wang, Sun, Cheng. Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis. ICML 2025.
[5] Zhang, Yue, Sun et al. Graph Sparsification via Mixture of Graphs. ICLR 2025 (Spotlight).
[6] Chen, Cheng, Amiri et al., Sun. FROG: Fair Removal on Graph. CIKM 2025.
[7] Liu, Sun, Meng et al. Nowhere to Hide: Online Rumor Detection Based on Retweeting Graph Neural Networks. IEEE TNNLS 2024.
[8] Sun, Yin, Liu et al. Structure Learning via Meta-Hyperedge for Dynamic Rumor Detection. IEEE TKDE 2022.
[9] Liu et al. ARC: A Generalist Graph Anomaly Detector with In-Context Learning. NeurIPS 2024.
[10] Niu et al. Zero-shot Generalist Graph Anomaly Detection with Unified Neighborhood Prompts. IJCAI 2025.
[11] Qiao et al. AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection. KDD 2025.
[12] Chen et al. AffinityTune: A Prompt-Tuning Framework for Few-Shot Anomaly Detection on Graphs. KDD 2025.
[13] GPCF. Neighbor-enhanced Graph Pre-training and Prompt Learning Framework for Fraud Detection. CIKM 2025.
[14] DGA-GNN. Dynamic Grouping Aggregation GNN for Fraud Detection. AAAI 2024.
[15] Neo et al. Towards Fair Graph Anomaly Detection: Problem, Benchmark Datasets, and Evaluation. CIKM 2024.
[16] Li et al. FPrompt: Fairness-aware Prompt Tuning for Graph Neural Networks. WWW 2025.
[17] Antonelli & Bojchevski. Data Valuation for Graphs. TMLR 2025.
[18] Sun, Zhang, Wu, Cheng, Xiong, Li. Graph Prompt Learning: A Comprehensive Survey and Beyond. arXiv 2023.
