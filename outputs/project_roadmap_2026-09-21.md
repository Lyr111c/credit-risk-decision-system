# 信用风险决策系统项目推进计划（中文）

编制日期：2026-09-21。适用仓库：`D:\Credit_Risk_Decision_System`。

本文是供 agent 分阶段执行的实施计划，不代表下列功能已经实现。默认目标是完成可复现、可解释的教学与作品集项目：从已有客户违约概率预测，推进到评分卡、模型比较、假设成本下的决策模拟和监控原型。默认本地运行，不要求云部署、在线授信服务或商业系统集成。

## 1. 已有基础与待完成范围

已检查 `README.md`、主要源代码、测试、Notebook 文件和已有报告。2026-09-21 实际执行 `.\.venv\Scripts\python.exe -m pytest -q`，结果为 **41 passed in 22.68s**。本次未重新执行真实数据训练，下面的模型数字引用已有报告。

| 已有部分 | 当前证据与状态 |
| --- | --- |
| 数据 | UCI 信用卡违约数据已在本地；报告记录 30,000 条、25 列，违约率 22.12%，无缺失单元格，去掉 ID 后有 35 条内容重复记录 |
| 数据审计 | `src/credit_risk/data.py` 提供下载、哈希校验、加载、字段规范化和双语报告生成 |
| EDA | `analysis.py` 提供变量汇总、还款状态分析、切分分布比较和图表生成 |
| 基线 | `modeling.py` 实现固定种子 42 的 60%/20%/20% 分层切分、训练集内预处理，以及 Dummy、普通 Logistic Regression、balanced Logistic Regression |
| 评估 | `evaluation.py` 提供 ROC-AUC、PR-AUC、KS、Brier Score、固定阈值指标和校准曲线；其中 `pr_auc` 实际使用 Average Precision，后续须明确标注 |
| 实验与报告 | `experiment.py`、3 个 Notebook、3 份双语报告和 10 张图表已存在 |
| 已有表现 | 普通逻辑回归测试集 ROC-AUC 0.7591、KS 0.3970、Brier Score 0.1388；不是本次重新训练结果 |

尚未实现：WOE/IV 评分卡、树模型 Challenger、概率校准器、成本驱动策略、监控原型及完整模型交付接口。

关键边界：数据描述已有客户行为，不能直接证明新客户贷前审批效果；没有可用的真实跨时期验证数据，随机测试集不是 OOT。已有测试集已被报告过，继续保留作为历史基准，不能重新称为“从未查看的测试集”；不能靠重新随机切分恢复独立性。

## 2. 执行约定与目录安排

- 不重写已完成模块，不大规模调整目录；优先新增独立模块和最小兼容修改。
- 现有 `reports/` 报告保留为历史基线。新增面向读者的交付物统一写入 `outputs/`；临时搜索结果、草稿和过程文件进入 `work/`。本计划不授权删除或覆盖已有成果。
- 建议结构：`configs/` 保存版本化实验配置；`outputs/reports/` 保存双语报告；`outputs/figures/` 保存图表；`outputs/artifacts/<run_id>/` 保存模型、策略和元数据；`outputs/predictions/` 保存本地预测结果。
- 原始数据、客户级预测、含客户 ID 的切分清单、序列化模型及临时文件不得意外进入 Git；阶段 0 增补精确的忽略规则，不把整个 `outputs/` 忽略，以免计划和报告无法版本管理。
- 所有新增或实质性更新的读者文档在同一文件内先完整中文、后完整英文；事实、指标、限制一致。图题和必要表头提供双语。自动报告必须通过生成器更新并重新生成。
- 默认串行执行。每个阶段只完成自己的任务和验收，不自行创建其他用户任务，不自动发布或部署。
- 所有数值结果来自真实执行。未运行的功能标为计划，失败结果保留原因，不以“预期通过”代替验证。

## 3. 总体顺序与完成门槛

`P0 实验边界 → P1 WOE/IV → P2 评分卡 → P3 Challenger → P4 校准与模型选择 → P5 策略选择 → P6 冻结评估 → P7 监控 → P8 推理交付 → P9 总体验收`

| 阶段 | 主要结果 | 依赖 | 完成门槛 |
| --- | --- | --- | --- |
| P0 | 配置、切分清单、开发/最终评估分离 | 现有基线 | 开发流程无法自动读取测试标签 |
| P1 | 可复用 WOE/IV 转换器 | P0 | 分箱和 IV 仅使用训练折标签，边界测试通过 |
| P2 | 可解释评分卡 | P1 | 分数与违约概率方向一致、可相互还原 |
| P3 | 一个树模型 Challenger | P0、P2 | 相同切分下完成预算内开发比较 |
| P4 | 校准方案与候选模型冻结 | P2、P3 | 校准不使用验证/测试标签拟合，选择规则有记录 |
| P5 | 教学成本下的策略 | P4 | 阈值在验证集选择，约束和无解状态明确 |
| P6 | 最终模型与策略报告 | P5 | 冻结后评估测试集，不根据结果重新选择 |
| P7 | 批次监控原型 | P6 | 支持无标签监控，模拟漂移明确标注 |
| P8 | 模型包与批量预测入口 | P6、P7 | 保存/加载前后输出一致，输入错误清晰 |
| P9 | 可复现项目交付 | P8 | 端到端复现、文档一致、测试通过 |

P3 技术上只依赖 P0，但默认在评分卡完成后启动，以控制同时变化的范围。P8 的输入契约可以提前设计，正式打包必须等待 P6 冻结。

## 4. P0：固定实验协议并保护评估边界

**工作步骤**

1. 检查 Git 状态、Python 和依赖版本、原始文件哈希；记录实际环境，保留用户未提交修改。将可复现环境快照保存到 `outputs/artifacts/<run_id>/environment.txt`，不要直接把现有版本范围当成精确锁定环境。
2. 创建 `configs/experiment_v2.json`，统一管理随机种子、数据哈希、特征清单、切分比例、CV 折数、模型搜索预算、评分映射、校准与策略规则；为配置建立字段校验。
3. 固定现有 60%/20%/20% 划分，保存客户 ID 到 split 的本地映射及其哈希；训练、验证、测试 ID 不重叠。ID 只用于关联与验证，不作为模型特征。
4. 修改 `experiment.py`：将目前自动访问测试集的逻辑拆成开发实验和显式最终评估。旧接口如需保留，作为文档明确的历史基线复现入口，新的搜索和 Notebook 不调用它。
5. 默认训练集内部使用 5 折分层交叉验证。所有需要学习数据分布或标签的步骤都放在折内 Pipeline；验证集用于候选选择和阈值选择；测试集只用于 P6 的冻结评估。
6. 标注 `pr_auc` 为 Average Precision（AP）；如新增梯形积分 PR-AUC，使用另一个字段，禁止同名混用。保持旧报告解释可追溯。
7. 明确 35 条非 ID 重复记录的保留规则，检查其是否跨切分，并在限制中报告。可做独立的分组切分敏感性实验，但不能混入主比较或偷偷改变基准。

**文件**：新增 `src/credit_risk/config.py`、`protocol.py`、`configs/experiment_v2.json`、`tests/test_protocol.py`；最小修改 `experiment.py`、`evaluation.py` 和 `.gitignore`。生成 `outputs/reports/experiment_protocol.md`。

**验收**：配置非法值被拒绝；切分可复现、互斥、完整；用会在访问时失败的测试标签对象或等效隔离测试证明开发入口不依赖测试标签；预处理折内拟合；旧 41 项测试无非预期回归。执行相关测试，再运行完整套件一次。

## 5. P1：实现分箱、WOE 与 IV

**工作步骤**

1. 新增符合 scikit-learn `fit/transform` 约定的转换器，输出稳定的特征名，支持克隆和 Pipeline；`transform` 不需要标签，也不能重新计算分箱。
2. 首版采用可审计的训练集分位数分箱：数值变量最多 5 箱，合并重复切点，常量变量为单箱；相邻小箱合并直至每箱至少占训练样本 5% 或仅剩一箱。分类变量低频水平按同样 5% 门槛合并到 OTHER；这些是教学默认值，必须配置化。
3. 缺失值进入独立箱；数值区间覆盖负无穷到正无穷，明确左右闭合规则。未见类别映射到训练期 OTHER，若不存在 OTHER 则使用中性 WOE=0 并记录回退。不得擅自解释来源未定义的还款状态编码。
4. 定义违约为 bad=1、未违约为 good=0。对 K 个训练期箱使用平滑参数 α=0.5：`p_good_j=(good_j+α)/(N_good+αK)`，`p_bad_j=(bad_j+α)/(N_bad+αK)`，`WOE_j=ln(p_good_j/p_bad_j)`，`IV=Σ(p_good_j-p_bad_j)×WOE_j`。缺失箱若属于训练期箱，计入 K；未见类别中性回退不进入训练 IV。
5. 输出每箱边界/类别、样本数、good/bad 数量、违约率、WOE、IV 分量和变量总 IV。IV 首版用于解释，不自动按固定阈值删除变量。任何后续 IV 筛选都必须置于 CV 折内。
6. 检查数值变量的 WOE 趋势；首版报告非单调情况，不盲目强制单调。最优分箱或单调合并属于后续可选增强。

**文件**：新增 `src/credit_risk/woe.py`、`tests/test_woe.py`；生成 `outputs/reports/woe_iv_report.md` 和对应机器可读分箱表。

**验收**：用手算数据核对平滑、WOE 和 IV；覆盖全 good/全 bad 箱、缺失、无穷值拒绝、常量、重复分位点、边界值、未见类别；整体单类别训练数据必须报错。改变验证/测试标签不能改变分箱或 IV。执行 `tests/test_woe.py` 和相关 Pipeline 测试。

## 6. P2：构建评分卡并解释分数

**工作步骤**

1. 创建 `WOETransformer → LogisticRegression` Pipeline，首版不使用类别平衡权重。使用训练集 CV 比较 `C ∈ {0.01, 0.1, 1, 10}`，以平均 ROC-AUC 为主，Brier 为辅助；平局优先更强正则化。现有普通逻辑回归保留为参照。
2. 定义 good:bad odds，采用教学映射 `base_score=600`、`base_odds=20`、`PDO=50`，并在报告明确它们不是行业统一标准。`B=PDO/ln(2)`，`A=base_score-B×ln(base_odds)`，`score=A-B×logit(PD)`，分数越高风险越低。
3. 对 WOE 逻辑回归 `logit(PD)=β0+Σβj×WOEj`，给出基础分 `A-Bβ0` 与每个变量分数 `-Bβj×WOEj`；保持内部浮点精度，只在展示时取整。概率反解采用 `PD=1/(1+exp((score-A)/B))`，实现时避免数值溢出。
4. 导出每箱分值、系数、总分和模型版本；提供相对训练期参考水平的主要减分因素，明确它们是模型解释而非因果结论或合规拒绝理由。
5. 在验证集比较普通逻辑回归和评分卡的 ROC-AUC、AP、KS、Brier、分数分布和分段违约率；不读取测试结果决定分箱。

**文件**：新增 `scorecard.py`、`tests/test_scorecard.py`、`notebooks/04_scorecard.ipynb`；生成 `outputs/reports/scorecard_report.md`、分箱分值表及图。

**验收**：good:bad odds 翻倍时分数增加 50；20:1 对应 600 分；总分等于基础分加各项贡献；分数可还原原始 PD；缺失和未见类别下仍可预测。注意后续校准会改变 PD，必须将原始评分卡分数与校准后决策 PD 分开记录，不能继续声称原有加性分数精确对应校准后 PD。

## 7. P3：增加一个树模型 Challenger

**工作步骤**

1. 首版采用现有 scikit-learn 依赖中的 `HistGradientBoostingClassifier`，暂不同时加入多种外部模型库。建立独立预处理器，使用稠密 One-Hot、缺失数值的训练折内中位数填补；保留字段分组，避免直接把未知类别编码当成具有确定间距的数值。
2. 所有候选使用相同训练/验证划分和 CV 折。搜索上限为 12 组参数×5 折：`learning_rate={0.05,0.1}`、`max_leaf_nodes={7,15,31}`、`l2_regularization={0,1}`，固定 `max_iter=200`、`early_stopping=False`、随机种子 42。资源不足时先记录并统一缩减预算，不根据测试结果增减搜索。
3. 训练集 CV 以 ROC-AUC 选择参数，平局以 Brier、复杂度排序；保存每折指标、耗时、警告、失败配置和实际版本。最终只在训练集拟合选定候选并输出验证预测。
4. 与 Dummy、普通逻辑回归和评分卡比较，类别平衡版本只作为已有敏感性参照。比较排序能力、概率误差和运行开销，不保证树模型一定优胜。
5. 在验证集计算置换重要性，说明相关变量可能分摊重要性；解释结果不用于事后无限加特征。本阶段不强制 SHAP 或复杂特征工程。

**文件**：新增 `challenger.py`、`selection.py`、`tests/test_challenger.py`；扩展配置；新增 `notebooks/05_model_comparison.ipynb`；生成 `outputs/reports/model_comparison_development.md`。

**验收**：搜索不读取测试集；预处理在 CV 内；模型可处理训练未见类别；重复运行切分和选定参数可复现；记录资源消耗。运行相关测试和一次预算受控的真实数据实验。

## 8. P4：概率校准与模型候选选择

**工作步骤**

1. 明确区分“校准曲线诊断”和“拟合校准器”。每个进入比较的普通逻辑回归、评分卡、Challenger 候选保留未校准版本，并建立 sigmoid 校准版本；isotonic 作为可选敏感性实验，不默认扩大搜索。
2. 使用训练集内部独立的 5 折分层校准流程，例如 `CalibratedClassifierCV` 的交叉验证方式；基模型预处理也必须在相应训练折拟合。验证集只评估校准结果，测试集不参与。记录实际 API、ensemble 行为和版本，基于安装版本实现并测试。
3. 比较验证集 ROC-AUC、AP、Brier 和校准图，图中显示每箱样本量；Log Loss 可作为新增辅助指标。小样本尾箱不作强结论。
4. 在运行前固定选择规则：先保留验证 ROC-AUC 距最优不超过 0.01 的候选，再优先最小 Brier；Brier 差小于 0.001 时优先较简单的候选，复杂度顺序写入配置。0.01 和 0.001 是教学容忍度，不是显著性检验阈值。
5. 冻结一个主模型和一个参照模型的结构、参数、校准方式与训练数据范围，记录选择理由。不得在 P6 看到测试结果后换主模型。

**文件**：新增 `calibration.py`、`tests/test_calibration.py`；扩展 `selection.py` 和比较 Notebook；生成 `outputs/reports/calibration_report.md`、`outputs/artifacts/<run_id>/model_selection.json`。

**验收**：能证明校准拟合不访问验证/测试标签；输出概率有限且在 [0,1]；基模型与校准模型名称不混淆；保存与重载前后概率一致。校准效果没有改善也属于有效结论，不能伪造提升。

## 9. P5：制定透明的成本场景与阈值策略

**工作步骤**

1. 新增 `configs/strategy_scenarios.json`。采用每个客户单位敞口的归一化教学场景，首个场景设净收益 `R=0.10`、违约净损失 `L=0.60`，并比较 `R={0.05,0.10,0.15}`、`L={0.30,0.60,0.90}` 的 9 种组合。这些不是数据估计或真实机构参数。
2. 明确收益表：批准未违约为 `+R`、批准违约为 `-L`、拒绝为 0。被拒优质客户的潜在收益已经包含在与批准方案的差值中，不重复额外扣罚。不要把信用额度当成真实放款额或 EAD。
3. 基于预测 PD 的批准期望收益为 `(1-PD)R-PD×L`；无其他约束时盈亏平衡 PD 为 `R/(R+L)`。用此公式做逻辑核对，不宣称校准模型与简化成本足以代表现实利润。
4. 在验证集比较“全部批准”“全部拒绝”和 PD 阈值策略。统一约定 `PD<t` 批准、`PD≥t` 拒绝。固定阈值网格为 `[0,1]` 内步长 0.005，外加全部批准/拒绝的显式策略；默认按验证集观察收益选最大值，平局优先更低批准率。
5. 输出批准率、批准组违约率、拒绝组优质客户数、每个申请人平均收益和预测期望收益；全部拒绝时批准组违约率为 NA，不写 0。真实标签计算的收益仍只是场景模拟收益。
6. 增加可选约束示例：批准率至少 30%、批准组违约率至多 10%；明确这同样是教学示例。若无阈值满足，返回不可行并报告原因，不偷偷放宽约束。
7. 冻结主场景、模型、阈值、边界规则及所有敏感性场景的策略；将收益曲线和约束可行区域写入报告。承认复用验证集进行模型和策略选择的选择偏差，最终报告用 P6 检查泛化。

**文件**：新增 `strategy.py`、`tests/test_strategy.py`、`notebooks/06_decision_strategy.ipynb`；生成 `outputs/reports/strategy_development_report.md` 与 `outputs/artifacts/<run_id>/policy.json`。

**验收**：手算小样本验证收益、批准方向、阈值等号、全批/全拒和无解情况；任何测试集信息不能改变冻结阈值；9 个场景结果均来自实际计算。

## 10. P6：冻结后的最终评估与不确定性

**工作步骤**

1. 在评估前保存冻结清单：数据/切分/配置哈希、代码版本及工作区状态、模型和校准参数、策略阈值、指标定义。禁止报告生成函数顺便重新调参。
2. 首版保持模型仅用原训练集及其内部 CV 拟合，不再合并训练与验证集重训，以免改变校准与已选阈值含义。若未来需要重训，必须作为新版本设计独立校准和评估协议。
3. 对冻结主模型、参照模型、Dummy 和固定策略运行最终测试评估；保存指标和必要的本地预测供绘图复用。重复绘图不重新拟合或选择。
4. 对 ROC-AUC、AP、KS、Brier、批准率、批准组违约率和模拟收益给出点估计；采用同一批样本索引的配对 bootstrap，默认 1,000 次、种子 42，报告 95% 百分位区间及主模型相对参照的差值区间。单类别重采样中未定义的指标跳过并报告有效次数；全拒时相关指标保持 NA。
5. 说明 bootstrap 反映固定模型和固定策略在当前样本上的抽样不确定性，不包含训练与模型选择的全部不确定性，也不证明跨时期稳定性。
6. 补充分组诊断：按性别、年龄段等报告样本量、违约率、批准率和可定义的误差指标；小组不足 100 条或少于 20 个正/负例时标记不稳定。不得据此宣称公平或合规；如考虑去除敏感属性，作为预先声明的独立敏感性实验，不用测试结果反向调参。
7. 若测试结果退化，照实写结论。只有发现实现错误才修复并记录重跑原因，不能为追求更高指标反复回到 P3 搜索。

**文件**：新增 `final_evaluation.py`、`uncertainty.py`、`tests/test_final_evaluation.py`、`tests/test_uncertainty.py`；生成 `outputs/reports/final_model_report.md`、`final_strategy_report.md`、`model_card.md`。

**验收**：最终命令只加载冻结配置/产物；bootstrap 指标差使用相同重采样索引；报告数字与机器结果一致；说明历史测试集已经看过以及不存在 OOT 的限制；双语数字、表格与警告一致。

## 11. P7：实现批次监控原型

**工作步骤**

1. 使用训练数据固定数值/分数分箱、类别集合和基准比例，保存监控基准。新批次不得重算基准边界；未知类别和缺失值单独计数。
2. 无标签时输出样本量、字段/类型异常、缺失率、未知类别率、特征 PSI、分数/PD 分布、批准率；有标签时再输出实际违约率、ROC-AUC、AP、Brier 和模拟收益。仅有批准客户标签时，注明选择性标签导致的评估限制。
3. 统一 PSI 公式 `Σ(q-p)ln(q/p)`，对 p、q 使用配置化平滑并重新归一化；报告分箱、平滑参数和样本量。示例阈值 0.10/0.25 仅为教学告警配置，不称为通用行业标准。
4. 对目标样本过少、单一目标类别和无标签批次返回明确状态，不产生误导性指标或崩溃。
5. 以随机分批作为稳定性演示，并单独构造带人工分布扰动的批次验证告警。显式标为“模拟批次/模拟漂移”，禁止按客户 ID 排序伪造时间轴。
6. 输出 Champion-Challenger 比较和人工复核/重新评估建议；监控告警不自动触发重训或替换模型。

**文件**：新增 `monitoring.py`、`tests/test_monitoring.py`、`configs/monitoring.json`、`notebooks/07_monitoring_demo.ipynb`；生成 `outputs/reports/monitoring_design.md`、`monitoring_demo_report.md`。

**验收**：相同分布 PSI 近零；已知扰动产生预期告警；零频箱、未知类别、无标签和小样本有测试；所有示例都清晰区分真实观测与人为模拟。

## 12. P8：模型打包与批量预测

**工作步骤**

1. 保存完整预处理 Pipeline、校准模型、可选原始评分卡、策略文件、分箱和监控基准，以及包含输入 schema、目标定义、依赖版本、数据/配置哈希和运行 ID 的 manifest。仅加载本项目可信来源的序列化模型。
2. 新增批量 CSV 推理入口。检查必需列、数据类型、缺失策略、重复 ID 和非法数值；允许声明的未知类别回退。目标列若存在，仅显式忽略，不进入特征；额外字段记录忽略清单。
3. 输出 `customer_id`（若提供）、`pd`、`decision`、`threshold`、`model_version`、`policy_version`；评分卡模型可额外输出 `raw_score` 和解释字段。无 ID 时保留输入行序并输出行号，不能假装生成真实客户身份。
4. PD 明确为最终校准或未校准概率；如有原始评分卡分数，记录其对应的原始 PD。只有实现一致的映射时才展示其他模型的分数，不能混称 WOE 评分卡。
5. 交付小型合成输入示例和操作说明，不在公开仓库放真实客户行。预测输出默认仅本地保存。

**文件**：新增 `artifacts.py`、`predict.py`、`tests/test_artifacts.py`、`tests/test_predict.py`；生成 `outputs/reports/inference_guide.md`、`outputs/examples/synthetic_applications.csv` 和本地模型包。

**验收**：模型保存/加载概率在指定容差内一致；行序和 ID 不错配；缺字段/错误类型明确报错；CLI 返回合理退出码；预测不需要真实目标标签。首版 CLI 足够，Web 页面/API 属于可选扩展。

## 13. P9：端到端复现、文档与交付

**工作步骤**

1. 汇总统一运行入口，按 `数据校验 → 开发实验 → 选择冻结 → 最终评估 → 监控示例 → 批量推理` 执行；区分开发模式和最终评估模式。
2. 运行全部测试；在实际可用的干净环境验证依赖安装和合成数据 smoke test，再使用本地原始数据完成一次正式流程。若网络或环境限制导致干净环境不可验证，明确记录，不伪称完成。
3. 执行新增 Notebook，保证调用 `src/` 而不是复制业务逻辑；导出到新版本路径，不覆盖历史基线。Notebook 不得在展示时自动触发重新调参或再次选择测试模型。
4. 核对报告均由实际产物生成，图表链接有效，中英内容一致，README 中的当前进度与命令匹配。保留历史基线结果并说明新版本的差异。
5. 记录每个 run 的耗时、依赖、配置和失败信息；检查 Git diff，避免原始数据、客户级结果、模型大文件、缓存和凭证进入版本库。
6. 形成最终交付索引，列出文件用途、运行顺序、关键结果、数据与评估限制、尚未验证事项。只总结实际完成内容。

**文件**：最小更新 `README.md`、`notebooks/README.md`、`tests/README.md`；新增 `outputs/reports/project_delivery.md` 和端到端 smoke test。若确需统一入口，新增 `src/credit_risk/cli.py`，避免多套相互矛盾的运行逻辑。

**验收**：完整测试通过；真实流水线可复现；所有最终指标可追溯到运行 ID；所有核心功能有可执行入口；未完成项逐条列出。不得把模拟利润、模拟漂移或随机测试表现写成生产收益和线上稳定性。

## 14. agent 每阶段执行模板

接手 agent 应按以下顺序执行，不把本计划中的未来文件或命令当成已存在：

1. 阅读根目录与目标目录的 `AGENTS.md`、本计划对应阶段、现有相关代码和测试。
2. 确认前置阶段产物存在且通过验收，检查工作区改动；给出本阶段简短计划。
3. 先实现最小可验证接口；对分箱、评估边界、评分映射、策略收益等高风险逻辑编写手算或隔离测试。
4. 实现功能并运行风险匹配的测试。真实数据实验只在合成测试通过后执行；不为通过测试而放宽定义。
5. 更新报告生成器和双语测试，生成真实报告与图表；不要只手改生成结果。
6. 对照阶段验收逐条核查。总结完成项、变更文件及用途、实际命令和结果、限制与下一阶段依赖。

建议任务指令：

> 按 `outputs/project_roadmap_2026-09-21.md` 实施阶段 Pn。先检查前置产物，遵守数据隔离、双语和目录要求；完成代码、必要测试、真实执行及报告生成。只实施本阶段；遇到无法验证的条件明确记录，不伪造结果，不因指标不够高而查看测试集调参。

## 15. 建议命令契约与验证清单

以下为拟实现的 CLI 契约，当前不能假定全部可运行；对应阶段必须实现并更新 `--help`。路径以仓库根目录为工作目录。

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
# P0–P4: development only; no test-label evaluation
.\.venv\Scripts\python.exe -m credit_risk.selection --config configs/experiment_v2.json
# P5: validation-only policy selection
.\.venv\Scripts\python.exe -m credit_risk.strategy --config configs/strategy_scenarios.json --run-id <run_id>
# P6: explicit evaluation of a frozen run
.\.venv\Scripts\python.exe -m credit_risk.final_evaluation --run-id <run_id>
# P7: clearly labeled simulated monitoring
.\.venv\Scripts\python.exe -m credit_risk.monitoring --run-id <run_id> --demo
# P8: batch inference from trusted local artifacts
.\.venv\Scripts\python.exe -m credit_risk.predict --artifact-dir outputs/artifacts/<run_id> --input outputs/examples/synthetic_applications.csv --output outputs/predictions/demo.csv
```

`<run_id>` 是占位符，执行时必须替换；配置内的默认输出路径、覆盖保护和运行状态由 P0/P8 实现。默认输出目录已存在时拒绝覆盖，复用已有结果须显式声明并校验哈希。冻结后的相同产物可以复现评估，但必须记录重跑且不得因此重新选择模型。

最终检查：

- [ ] 原始数据哈希、切分、环境和配置均可追溯。
- [ ] 分箱、编码、缺失填补、筛选、调参和校准均不使用测试标签。
- [ ] 开发流程不会自动执行测试评估。
- [ ] AP、WOE、odds、评分方向、阈值边界和收益定义一致。
- [ ] 模型与策略在最终评估前冻结。
- [ ] 全拒、无解、未知类别、无标签和单类别均有明确行为。
- [ ] 模拟假设、历史测试集已被查看和非 OOT 限制写入最终报告。
- [ ] 保存/加载和批量预测验证通过。
- [ ] 双语报告数字一致，图表可追溯，真实结果未被手工替换。

## 16. 优先级和可选扩展

最小核心交付为 P0–P6：形成“概率预测—解释—策略—冻结评估”的完整实验。完整项目需继续完成 P7–P9，提供监控、推理和复现交付。

完成核心验收后再考虑：受控特征工程、单调最优分箱、外部树模型、SHAP、交互式展示、API、真实时间数据/OOT 或外部验证。它们不是首版完成的前提；外部数据可用性、授权和质量需单独核实，不能承诺获取。无需为了执行本计划先向用户询问模型库或教学成本参数；默认值已给出，后续用户明确调整时再更新配置和协议。

本次仅新增推进计划，不实现计划中的功能，不改动现有代码或已有报告。

---

# Credit Risk Decision System Roadmap (English)

Prepared on 2026-09-21. Repository: `D:\Credit_Risk_Decision_System`.

This is a staged implementation plan for an agent, not a claim that the planned features already exist. The default objective is a reproducible, interpretable educational and portfolio project: extend existing-customer default prediction into scorecards, model comparison, decision simulation under hypothetical costs, and a monitoring prototype. Local execution is sufficient; cloud deployment, online underwriting, and commercial integration are outside the default scope.

## 1. Existing foundation and remaining scope

The README, main source modules, tests, notebook files, and existing reports were inspected. On 2026-09-21, `.\.venv\Scripts\python.exe -m pytest -q` actually returned **41 passed in 22.68s**. Real-data training was not rerun during this review; model numbers below come from the existing report.

| Existing component | Evidence and status |
| --- | --- |
| Data | The UCI credit-card default data are present locally; the report records 30,000 rows, 25 columns, a 22.12% default rate, no missing cells, and 35 duplicate-content records after excluding ID |
| Audit | `src/credit_risk/data.py` provides download, hash checks, loading, column normalization, and bilingual report generation |
| EDA | `analysis.py` provides feature summaries, repayment-status analysis, split-distribution comparisons, and figures |
| Baselines | `modeling.py` provides seed-42 stratified 60%/20%/20% splitting, training-only preprocessing, and Dummy, ordinary Logistic Regression, and balanced Logistic Regression |
| Evaluation | `evaluation.py` provides ROC-AUC, PR-AUC, KS, Brier Score, fixed-threshold metrics, and calibration curves; `pr_auc` actually uses Average Precision and must be labeled explicitly |
| Experiments and reports | `experiment.py`, 3 notebooks, 3 bilingual reports, and 10 figures exist |
| Reported performance | Ordinary logistic regression has test ROC-AUC 0.7591, KS 0.3970, and Brier Score 0.1388; these were not recomputed in this review |

Not yet implemented: WOE/IV scorecards, a tree Challenger, probability calibrators, cost-driven policies, a monitoring prototype, and a complete model-delivery interface.

Important boundaries: the data describe existing customers and cannot directly establish new-customer underwriting performance. No genuine temporal validation data are available, and the random test set is not OOT. Test results have already been reported: retain this split as a historical benchmark, never call it an untouched holdout, and do not claim that random resplitting restores independence.

## 2. Execution conventions and directories

- Preserve completed modules and the directory layout. Prefer independent additions and minimal compatible changes.
- Retain existing `reports/` documents as historical baselines. Put new reader deliverables in `outputs/`, and temporary searches, drafts, and intermediate files in `work/`. This plan does not authorize deleting or overwriting existing results.
- Suggested layout: `configs/` for versioned experiment settings; `outputs/reports/` for bilingual reports; `outputs/figures/` for figures; `outputs/artifacts/<run_id>/` for models, policies, and metadata; `outputs/predictions/` for local predictions.
- Prevent raw data, customer-level predictions, ID-bearing split manifests, serialized models, and temporary files from accidentally entering Git. P0 adds targeted ignore rules; do not ignore all of `outputs/`, because plans and reports should remain versionable.
- Every new or materially updated reader document must contain complete Chinese followed by complete English in one file, with identical facts, metrics, and limitations. Provide bilingual figure titles and necessary table headings. Update generators and regenerate automated reports.
- Execute sequentially by default. Complete each stage and its acceptance checks without creating other user tasks, publishing, or deploying automatically.
- All numerical results must come from actual execution. Label unrun features as planned, preserve failures and their causes, and never substitute expected success for verification.

## 3. Sequence and completion gates

`P0 Experiment boundaries → P1 WOE/IV → P2 Scorecard → P3 Challenger → P4 Calibration and model selection → P5 Policy selection → P6 Frozen evaluation → P7 Monitoring → P8 Inference delivery → P9 Final acceptance`

| Stage | Main result | Dependency | Completion gate |
| --- | --- | --- | --- |
| P0 | Configuration, split manifest, separate development/final evaluation | Existing baseline | Development cannot automatically access test labels |
| P1 | Reusable WOE/IV transformer | P0 | Binning and IV use training-fold labels only; edge tests pass |
| P2 | Interpretable scorecard | P1 | Scores have the correct risk direction and invert to probabilities |
| P3 | One tree Challenger | P0, P2 | Budgeted development comparison uses identical splits |
| P4 | Calibration and frozen model candidates | P2, P3 | Calibration never fits on validation/test labels; selection is recorded |
| P5 | Policy under educational costs | P4 | Thresholds use validation data; constraints and infeasibility are explicit |
| P6 | Final model and policy reports | P5 | Test evaluation follows freezing and does not trigger reselection |
| P7 | Batch monitoring prototype | P6 | Unlabeled monitoring works; simulated drift is labeled |
| P8 | Model bundle and batch inference | P6, P7 | Reloaded outputs agree; input errors are clear |
| P9 | Reproducible delivery | P8 | End-to-end reproduction, consistent documentation, passing tests |

P3 technically requires only P0, but starts after the scorecard by default to limit simultaneous changes. The P8 input contract can be designed early; final packaging waits for P6.

## 4. P0: Fix the experiment protocol and evaluation boundaries

**Steps**

1. Inspect Git status, Python/dependency versions, and raw-file hashes; record the actual environment and preserve user changes. Save a reproducible environment snapshot to `outputs/artifacts/<run_id>/environment.txt`; dependency ranges are not an exact environment lock.
2. Create `configs/experiment_v2.json` for seeds, data hashes, features, split proportions, CV folds, search budgets, score mapping, calibration, and policy rules, with field validation.
3. Preserve the existing 60%/20%/20% split. Save a local customer-ID-to-split mapping and hash; train, validation, and test IDs must be disjoint. IDs are for joins and verification only, never model features.
4. Split `experiment.py` into development experiments and explicit final evaluation, removing automatic test access from the development path. If retained, document the old interface as historical-baseline reproduction; new searches and notebooks must not call it.
5. Default to 5-fold stratified CV within training data. Every operation that learns distributions or labels belongs inside the fold Pipeline. Validation selects candidates and thresholds; test data are reserved for P6 frozen evaluation.
6. Label `pr_auc` as Average Precision (AP). If trapezoidal PR-AUC is added, use a different field. Keep historical report meanings traceable.
7. Document retention of the 35 non-ID duplicate-content rows, check whether they cross splits, and report the limitation. A separate grouped-split sensitivity experiment is optional; do not mix it into the primary comparison or silently change the benchmark.

**Files**: add `src/credit_risk/config.py`, `protocol.py`, `configs/experiment_v2.json`, and `tests/test_protocol.py`; minimally modify `experiment.py`, `evaluation.py`, and `.gitignore`. Generate `outputs/reports/experiment_protocol.md`.

**Acceptance**: reject invalid configuration; reproducible, disjoint, exhaustive splits; prove development does not depend on test labels using a failing-on-access label object or equivalent isolation test; preprocessing fits inside folds; no unintended regression in the existing 41 tests. Run focused tests, then the full suite once.

## 5. P1: Implement binning, WOE, and IV

**Steps**

1. Add a scikit-learn-compatible `fit/transform` transformer with stable feature names, cloning, and Pipeline support. `transform` requires no labels and never relearns bins.
2. Start with auditable training-quantile binning: at most 5 numeric bins, merge duplicate cut points, and use one bin for constants. Merge adjacent small bins until each contains at least 5% of training observations or only one remains. Pool rare categorical levels into OTHER using the same 5% threshold. These educational defaults must be configurable.
3. Give missing values a separate bin. Numeric intervals cover negative to positive infinity with explicit endpoint rules. Map unseen categories to a training OTHER bin, or use neutral WOE=0 and record the fallback if no OTHER exists. Do not invent meanings for source-undefined repayment codes.
4. Define bad=1 and good=0. For K training bins and smoothing α=0.5: `p_good_j=(good_j+α)/(N_good+αK)`, `p_bad_j=(bad_j+α)/(N_bad+αK)`, `WOE_j=ln(p_good_j/p_bad_j)`, and `IV=Σ(p_good_j-p_bad_j)×WOE_j`. A training missing bin counts toward K; neutral unseen-category fallback does not enter training IV.
5. Export boundaries/categories, counts, good/bad counts, default rates, WOE, IV contributions, and total feature IV. Use IV for explanation initially, not automatic fixed-threshold removal. Any later IV selection must occur within CV folds.
6. Inspect numeric WOE trends. Report non-monotonicity initially rather than forcing it blindly. Optimal binning and monotonic merging are optional later improvements.

**Files**: add `src/credit_risk/woe.py` and `tests/test_woe.py`; generate `outputs/reports/woe_iv_report.md` and machine-readable bin tables.

**Acceptance**: hand-check smoothing, WOE, and IV; cover all-good/all-bad bins, missing values, rejected infinite inputs, constants, repeated quantiles, endpoints, and unseen categories. Reject single-class training datasets. Changing validation/test labels cannot alter bins or IV. Run `tests/test_woe.py` and relevant Pipeline tests.

## 6. P2: Build and explain the scorecard

**Steps**

1. Build `WOETransformer → LogisticRegression` without class balancing initially. Compare `C ∈ {0.01, 0.1, 1, 10}` using training CV, primarily mean ROC-AUC and secondarily Brier; prefer stronger regularization for ties. Retain ordinary logistic regression as a reference.
2. Define good:bad odds and educational mapping `base_score=600`, `base_odds=20`, `PDO=50`, explicitly not universal industry standards. Set `B=PDO/ln(2)`, `A=base_score-B×ln(base_odds)`, and `score=A-B×logit(PD)`. Higher scores mean lower risk.
3. For `logit(PD)=β0+Σβj×WOEj`, report base points `A-Bβ0` and feature points `-Bβj×WOEj`. Preserve floating-point precision internally; round only for display. Invert with `PD=1/(1+exp((score-A)/B))`, using numerically stable implementation.
4. Export bin points, coefficients, total scores, and model versions. Explain major point reductions relative to training reference levels, labeling these as model explanations rather than causal claims or compliant adverse-action reasons.
5. Compare ordinary logistic regression and the scorecard on validation ROC-AUC, AP, KS, Brier, score distributions, and default rates by score band. Do not consult test results to change bins.

**Files**: add `scorecard.py`, `tests/test_scorecard.py`, and `notebooks/04_scorecard.ipynb`; generate `outputs/reports/scorecard_report.md`, bin-point tables, and figures.

**Acceptance**: doubling good:bad odds adds 50 points; 20:1 maps to 600; total points equal base plus contributions; scores invert to original PD; missing and unseen categories remain predictable. Later calibration changes PD: record raw scorecard scores separately from calibrated decision PD, and never claim the original additive score still exactly maps to calibrated PD.

## 7. P3: Add one tree Challenger

**Steps**

1. Initially use `HistGradientBoostingClassifier` from the existing scikit-learn dependency rather than adding several external libraries. Build a separate preprocessor with dense One-Hot and training-fold median imputation for numeric missing values. Preserve feature groups; do not treat undocumented category codes as known numerical distances.
2. Use identical training/validation partitions and CV folds for every candidate. Cap search at 12 configurations×5 folds: `learning_rate={0.05,0.1}`, `max_leaf_nodes={7,15,31}`, and `l2_regularization={0,1}`, fixing `max_iter=200`, `early_stopping=False`, and seed 42. If resources are insufficient, document a uniform budget reduction rather than changing search based on test results.
3. Select parameters by training-CV ROC-AUC, then Brier and complexity for ties. Save fold metrics, runtime, warnings, failed configurations, and actual versions. Fit the selected candidate only on training data and produce validation predictions.
4. Compare with Dummy, ordinary logistic regression, and the scorecard; the balanced baseline remains a sensitivity reference. Compare ranking, probability error, and runtime without promising that trees will win.
5. Compute validation permutation importance, explaining that correlated features may share importance. Do not use explanations to justify unbounded post-hoc feature searching. SHAP and complex feature engineering are not required here.

**Files**: add `challenger.py`, `selection.py`, and `tests/test_challenger.py`; extend configuration; add `notebooks/05_model_comparison.ipynb`; generate `outputs/reports/model_comparison_development.md`.

**Acceptance**: no test access during search; fold-local preprocessing; unseen-category support; reproducible splits and selected parameters; recorded resource usage. Run focused tests and one budgeted real-data experiment.

## 8. P4: Probability calibration and candidate selection

**Steps**

1. Distinguish calibration diagnostics from fitting a calibrator. Retain uncalibrated ordinary logistic, scorecard, and Challenger candidates and add sigmoid-calibrated versions. Isotonic is optional sensitivity analysis, not automatic search expansion.
2. Use a separate 5-fold stratified calibration procedure inside training data, such as the CV mode of `CalibratedClassifierCV`. Base-model preprocessing must fit within its training fold. Validation evaluates calibration only; test data do not participate. Record the actual API, ensemble behavior, and package version, implementing and testing against the installed version.
3. Compare validation ROC-AUC, AP, Brier, and calibration plots with bin counts. Log Loss may be added as a supporting metric. Do not overinterpret sparse tail bins.
4. Fix the rule before execution: retain candidates within 0.01 of the best validation ROC-AUC, then minimize Brier. If Brier differs by less than 0.001, prefer simpler candidates using a configured complexity order. These are educational tolerances, not significance-test thresholds.
5. Freeze one primary model and one reference, including architecture, parameters, calibration method, and training scope. Record the rationale and never replace the primary model after viewing P6 results.

**Files**: add `calibration.py` and `tests/test_calibration.py`; extend `selection.py` and the comparison notebook; generate `outputs/reports/calibration_report.md` and `outputs/artifacts/<run_id>/model_selection.json`.

**Acceptance**: demonstrate that calibration fitting never accesses validation/test labels; finite probabilities within [0,1]; distinct base/calibrated names; matching probabilities after reload. Lack of calibration improvement is a valid result, not something to conceal.

## 9. P5: Transparent cost scenarios and threshold policies

**Steps**

1. Add `configs/strategy_scenarios.json`. Use normalized educational unit exposure per customer, with initial net return `R=0.10` and net default loss `L=0.60`. Compare the 9 combinations of `R={0.05,0.10,0.15}` and `L={0.30,0.60,0.90}`. These are not estimated or real institutional parameters.
2. Define payoffs: approved non-default `+R`, approved default `-L`, rejected 0. Foregone return on rejected good customers is already reflected in comparisons with approval; do not subtract it again. Do not equate credit limits with actual loan amounts or EAD.
3. Expected approval payoff is `(1-PD)R-PD×L`; without other constraints the break-even PD is `R/(R+L)`. Use this as a logic check, not evidence that calibrated predictions and simplified costs establish real profits.
4. Compare approve-all, reject-all, and PD-threshold policies on validation data. Approve if `PD<t`, reject if `PD≥t`. Use a fixed threshold grid from 0 to 1 in steps of 0.005 plus explicit approve-all/reject-all policies. Maximize observed validation payoff by default; prefer lower approval rates for ties.
5. Export approval rate, default rate among approved customers, rejected good-customer count, average payoff per applicant, and predicted expected payoff. For reject-all, approved default rate is NA, not zero. Payoffs computed from observed labels remain hypothetical scenario returns.
6. Add an optional example constraint: approval rate at least 30% and approved default rate at most 10%, also labeled educational. Return infeasible with a reason if no threshold satisfies it; never silently relax constraints.
7. Freeze the primary scenario, model, threshold, boundary convention, and policies for all sensitivity scenarios. Report payoff curves and feasible regions. Acknowledge selection bias from using validation for both model and policy selection; assess generalization in P6.

**Files**: add `strategy.py`, `tests/test_strategy.py`, and `notebooks/06_decision_strategy.ipynb`; generate `outputs/reports/strategy_development_report.md` and `outputs/artifacts/<run_id>/policy.json`.

**Acceptance**: hand-check payoff, approval direction, threshold equality, approve-all/reject-all, and infeasibility; test information cannot change frozen thresholds; all 9 scenarios use actual calculations.

## 10. P6: Frozen final evaluation and uncertainty

**Steps**

1. Before evaluation, save data/split/config hashes, code revision and workspace status, model/calibration parameters, thresholds, and metric definitions. Report generation must not retune models.
2. Initially keep fitting restricted to the original training set and its internal CV. Do not refit on merged training and validation data, which would change calibration and threshold meaning. Future refitting requires a new version with a separate calibration/evaluation protocol.
3. Evaluate the frozen primary model, reference, Dummy, and fixed policies on test data. Save metrics and necessary local predictions for reusable plotting. Redrawing figures does not refit or reselect.
4. Report point estimates for ROC-AUC, AP, KS, Brier, approval rate, approved default rate, and simulated payoff. Use paired bootstrap with shared sample indices, default 1,000 repetitions and seed 42, reporting 95% percentile intervals and primary-minus-reference difference intervals. Skip undefined metrics in single-class resamples and report valid counts; preserve NA for reject-all metrics.
5. Explain that bootstrap captures sampling uncertainty for fixed models/policies on the current sample, not all training/selection uncertainty or temporal stability.
6. Add subgroup diagnostics by sex, age band, and similar fields: counts, default rates, approval rates, and defined error metrics. Flag instability below 100 observations or fewer than 20 positive/negative cases. Do not claim fairness or compliance. Removing sensitive attributes must be a separately predeclared sensitivity experiment, not a reaction to test results.
7. Report deteriorating test results honestly. Fix implementation bugs with documented rerun reasons; do not repeatedly return to P3 to chase better test metrics.

**Files**: add `final_evaluation.py`, `uncertainty.py`, `tests/test_final_evaluation.py`, and `tests/test_uncertainty.py`; generate `outputs/reports/final_model_report.md`, `final_strategy_report.md`, and `model_card.md`.

**Acceptance**: the final command loads only frozen configuration/artifacts; bootstrap differences share resample indices; reported numbers match machine outputs; acknowledge previously viewed historical test data and no OOT; bilingual numbers, tables, and warnings agree.

## 11. P7: Batch monitoring prototype

**Steps**

1. Freeze numeric/score bins, category sets, and reference proportions using training data, and save the baseline. New batches must not redefine edges. Count unknown categories and missing values separately.
2. Without labels, report volume, schema/type errors, missingness, unknown-category rates, feature PSI, score/PD distributions, and approval rate. With labels, add observed default rate, ROC-AUC, AP, Brier, and simulated payoff. If labels cover approved customers only, state selective-label limitations.
3. Use `PSI=Σ(q-p)ln(q/p)`, with configured smoothing and renormalization for p and q. Report bins, smoothing, and sample sizes. Example alert levels 0.10/0.25 are educational settings, not universal industry standards.
4. Return explicit statuses for undersized, single-class, and unlabeled batches instead of misleading metrics or crashes.
5. Use random batches for a stability demonstration and separately inject artificial distribution shifts to test alerts. Label them simulated batches/drift. Never sort customer IDs to invent a time axis.
6. Produce Champion-Challenger comparisons and recommendations for manual review/re-evaluation. Alerts must not automatically retrain or replace models.

**Files**: add `monitoring.py`, `tests/test_monitoring.py`, `configs/monitoring.json`, and `notebooks/07_monitoring_demo.ipynb`; generate `outputs/reports/monitoring_design.md` and `monitoring_demo_report.md`.

**Acceptance**: identical distributions yield near-zero PSI; known perturbations produce expected alerts; tests cover zero-frequency bins, unknown categories, unlabeled data, and small samples; examples distinguish observations from simulations.

## 12. P8: Model packaging and batch prediction

**Steps**

1. Save the complete preprocessing Pipeline, calibrated model, optional raw scorecard, policy, bins, and monitoring baseline, plus a manifest with input schema, target definition, dependencies, data/config hashes, and run ID. Load serialized models only from trusted project sources.
2. Add a CSV batch inference entry point. Validate required fields, types, missing-value policy, duplicate IDs, and invalid numbers, while allowing documented unknown-category fallback. Explicitly ignore a supplied target column; it never becomes a feature. Record ignored extra fields.
3. Output `customer_id` if supplied, `pd`, `decision`, `threshold`, `model_version`, and `policy_version`. Scorecards may also emit `raw_score` and explanations. Without ID, preserve input order and output row numbers without pretending they are real customer identities.
4. Label PD as the final calibrated or uncalibrated probability. Record the original PD associated with any raw scorecard score. Show scores for other models only with a consistent mapping; do not call them WOE scorecards.
5. Deliver a small synthetic input example and usage guide, never real customer rows in the public repository. Predictions remain local by default.

**Files**: add `artifacts.py`, `predict.py`, `tests/test_artifacts.py`, and `tests/test_predict.py`; generate `outputs/reports/inference_guide.md`, `outputs/examples/synthetic_applications.csv`, and a local model bundle.

**Acceptance**: saved/reloaded probabilities match within a specified tolerance; IDs and row order remain aligned; missing fields and wrong types fail clearly; CLI exit codes are meaningful; prediction requires no labels. A CLI is sufficient initially; web/API interfaces are optional.

## 13. P9: End-to-end reproduction, documentation, and delivery

**Steps**

1. Consolidate execution into `data verification → development → selection freeze → final evaluation → monitoring demo → batch inference`, with distinct development and final-evaluation modes.
2. Run all tests. Verify dependency installation and a synthetic-data smoke test in an actually available clean environment, then run the full process once with local raw data. Disclose network/environment restrictions if clean-environment verification is impossible.
3. Execute new notebooks that call `src/` rather than duplicate business logic; write versioned outputs without overwriting historical baselines. Displaying notebooks must not automatically retune or reselect test models.
4. Verify report provenance, figure links, bilingual consistency, and README status/commands. Preserve historical results and explain differences in the new version.
5. Record runtime, dependencies, configuration, and failures per run; inspect Git diff to exclude raw data, customer-level outputs, large models, caches, and credentials.
6. Create a delivery index covering file purposes, execution order, findings, data/evaluation limitations, and unverified items. Summarize only completed work.

**Files**: minimally update `README.md`, `notebooks/README.md`, and `tests/README.md`; add `outputs/reports/project_delivery.md` and an end-to-end smoke test. Add `src/credit_risk/cli.py` only if consolidation needs it; avoid conflicting execution paths.

**Acceptance**: passing full suite; reproducible real-data workflow; final metrics traceable to run IDs; executable interfaces for core features; explicit unfinished items. Never describe simulated returns/drift or random-test performance as production profit or online stability.

## 14. Per-stage agent execution template

The implementing agent must follow this sequence and must not assume planned files or commands already exist:

1. Read root/target `AGENTS.md`, the relevant roadmap stage, and existing code/tests.
2. Verify prerequisite artifacts and acceptance, inspect workspace changes, and provide a short stage plan.
3. Implement the smallest verifiable interface. Use hand-checkable or isolation tests for risky binning, evaluation boundaries, score mapping, and policy payoff logic.
4. Implement and run risk-appropriate tests. Run real-data experiments after synthetic tests pass; never weaken definitions to make tests pass.
5. Update report generators and bilingual tests, then generate actual reports/figures. Do not only hand-edit generated outputs.
6. Check every acceptance item. Report completed work, changed files and purposes, actual commands/results, limitations, and dependencies for the next stage.

Suggested task instruction:

> Implement stage Pn from `outputs/project_roadmap_2026-09-21.md`. Check prerequisites first, obey data-isolation, bilingual, and directory requirements, and complete code, necessary tests, actual execution, and report generation. Implement only this stage; document unverifiable conditions, do not fabricate results, and do not tune on test data to improve disappointing metrics.

## 15. Proposed command contract and verification checklist

These are planned CLI contracts, not commands assumed to work already. Each owning stage must implement them and update `--help`. Run from the repository root.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
# P0–P4: development only; no test-label evaluation
.\.venv\Scripts\python.exe -m credit_risk.selection --config configs/experiment_v2.json
# P5: validation-only policy selection
.\.venv\Scripts\python.exe -m credit_risk.strategy --config configs/strategy_scenarios.json --run-id <run_id>
# P6: explicit evaluation of a frozen run
.\.venv\Scripts\python.exe -m credit_risk.final_evaluation --run-id <run_id>
# P7: clearly labeled simulated monitoring
.\.venv\Scripts\python.exe -m credit_risk.monitoring --run-id <run_id> --demo
# P8: batch inference from trusted local artifacts
.\.venv\Scripts\python.exe -m credit_risk.predict --artifact-dir outputs/artifacts/<run_id> --input outputs/examples/synthetic_applications.csv --output outputs/predictions/demo.csv
```

Replace `<run_id>` before execution. P0/P8 implement configured output defaults, overwrite protection, and run state. Reject existing output directories by default; explicit reuse requires hash verification. Identical frozen artifacts may be reevaluated for reproduction, but reruns must be recorded and cannot trigger model reselection.

Final checks:

- [ ] Raw hashes, splits, environment, and configuration are traceable.
- [ ] Binning, encoding, imputation, selection, tuning, and calibration never use test labels.
- [ ] Development does not automatically evaluate test data.
- [ ] AP, WOE, odds, score direction, threshold boundaries, and payoff definitions agree.
- [ ] Models and policies are frozen before final evaluation.
- [ ] Reject-all, infeasibility, unknown categories, missing labels, and single-class cases are defined.
- [ ] Reports disclose simulated assumptions, previously viewed test data, and no OOT.
- [ ] Serialization and batch inference verification pass.
- [ ] Bilingual numbers agree, figures are traceable, and results were not manually substituted.

## 16. Priorities and optional extensions

The minimum core delivery is P0–P6, completing prediction, explanation, policy, and frozen evaluation. The complete project continues through P7–P9 for monitoring, inference, and reproducible delivery.

After core acceptance, consider controlled feature engineering, monotonic optimal binning, external tree libraries, SHAP, interactive presentation, APIs, genuine temporal/OOT data, or external validation. None is a prerequisite for the initial version. External data availability, authorization, and quality require separate verification and cannot be promised. No advance user choice of model library or teaching cost parameters is needed to execute this plan: defaults are provided and can be revised through configuration/protocol updates if the user requests changes.

This task only adds the roadmap; it does not implement the planned features or modify existing code or reports.
