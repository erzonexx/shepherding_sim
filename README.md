# Dual-Layer Autonomous Swarm Shepherding

异质羊群牧羊仿真：规则层逐帧控制狗，LLM在异常事件触发时提出模式或目标建议。
当前主线模型为本地优化 **Qwen3.6 NVFP4＋MTP3**。
**LLM选择修复模式的研究已告一段落**：已知异常A/B/C下，任务完成优先MPC组合在独立400场中优于规则，但固定启发式＋MPC与LLM＋MPC无显著差异，LLM的额外贡献未得到证实。结论见下文“LLM选择修复模式的结论”，完整记录见[EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md)。

## 常用入口

在项目根目录运行，使用上级目录中的 `.venv` Python环境。

```powershell
# 交互查看纯规则仿真
python main.py --controller rule --scenario stress --seed 70 --no-video

# 批量实验（普通批处理，不等同于按真实时钟运行）
python run_experiments.py --help

# 纯规则30/10Hz与六场并行测试（--run才会开始实际运行）
python experiments/rule_timing.py --help

# 10Hz双层仿真1/2/4/6并发延迟测试（使用同一份Spark高速模型）
python experiments/qwen_parallel_timing.py --help

# 三候选MPC固定六场小测（已完成批次拒绝覆盖；数据保存在logs）
python experiments/mpc_pilot.py --help

# 已有MPC输入的单/双请求及连接复用计时（已完成批次拒绝覆盖）
python experiments/mpc_latency_diagnostic.py --help

# MPC输入准备优化的离线前后对照（已完成批次拒绝覆盖）
python experiments/mpc_preparation_timing.py --help

# 纯规则/原双层/单候选MPC/双候选MPC四组六场机制对照
python experiments/mpc_candidate_comparison.py --help

# 纯规则与原单候选MPC的新100场配对验证（--run启动，历史批次不可覆盖）
python experiments/mpc_validation.py --help

# 离线复盘新100场中的救回/退步，不调用模型
python experiments/mpc_horizon_review.py --help

# 任务完成优先MPC：新200场探索、独立400场确认、五组100场贡献拆分（均已完成，历史批次不可覆盖）
python experiments/mpc_completion_validation.py --help
python experiments/mpc_confirmation.py --help
python experiments/mpc_component_validation.py --help

# A类专用oracle难度标定（纯规则，不调用模型；正式对照未执行）
python experiments/oracle_permissions.py --help

# 回归测试，不调用真实在线模型
python -B -m unittest discover -s tests
```

本地Qwen的模型身份、端口与Sync启停见[SPARK_SETUP.md](SPARK_SETUP.md)。
后续新对照实验默认采用 **10Hz、每组最多六场并行**，沿用后台调用与连续控制机制。
计划同时运行六场纯规则和六场双层，共十二场；首次合开前先核验共同负载下的控制计时和内存。
两组使用相同种子、组成及帧数预算；保留现有逐帧参数，30帧介入在10Hz下对应3秒。
这项约定用于后续新实验，未全局重写旧入口；历史30Hz协议和冻结证据保持原样。
历史批次脚本不能直接当作新的实验入口。

## 文件放在哪里

```text
shepherding_sim/
├── main.py / run_experiments.py   交互与批量入口
├── config.py                     参数
├── environment.py / dynamics.py   环境与运动模型
├── detector.py                   报警
├── *_controller.py               规则、高层与后台连续控制
├── mode_forecast.py               短期预测
├── *_backend.py / vllm_*input.py  模型调用与输入格式
├── experiments/                  条件实验与纯规则节奏/并行测试
├── tests/                        自动回归测试
├── tools/                        分析与实时实验工具
│   └── spark/                    模型启停脚本源码
├── docs/                         完整技术说明与早期实验记录
├── logs/                         实验数据与冻结证据（不入Git）
└── local_artifacts/               本地文档材料（不入Git）
```

根目录运行模块目前都有引用；本轮没有通过兼容包装文件增加新的入口层。
历史临时脚本已移到项目旁的 `shepherding_sim_archive/cleanup_20261020/`，
原位置和哈希可在该目录的 `organization_manifest.json` 中查到。

## 继续研究前看什么

- [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md)：已有结果与待做计划。
- [完整技术说明](docs/technical_reference.md)：模式、接口、参数和命令细节。
- [条件实验说明](experiments/anomaly_behavior/README.md)：各类历史试验。
- [早期实验记录](docs/early_experiments.md)：原 `tools/experiment_notes.md`。

MPC试验在双警报触发时比较 **动态规则方案＋两个LLM方案**；同一份高速模型并发提出模式和目标。
回复到达后用最新状态预测60帧，最多介入30帧，包含实际提前释放和交还；重复、取消或失效的建议不接管。
首次六场三组均为5/6成功，MPC实际未接管LLM建议；完整延迟、候选重复与评分取舍见实验记录。
后续四组六场小测也均为5/6；单候选MPC实际接管两次，配对救回1场、退步1场；双候选未接管。
单候选与双候选已交付轮次的平均总耗时约1.96秒、2.97秒，样本不同，尚无稳定成功率优势。
预测窗口有效性、候选多样性及评分仍需验证，不自动扩大测试。

历史简化对照保留双警报触发，允许已发出的建议在报警变化后继续进入最新状态预测，并去掉评分之前的额外逐项风险门槛；原评分和时效检查保留。
同六场中，规则5/6、原单候选MPC和简化版均6/6，平均封顶帧数分别为1114.33、1064.33、842.33。
原版取消7/14轮，简化版0/9轮；仅为已知场景的组合机制小测，不能据此判断稳定优势。默认路径仍为原版。

随后用20001–20050共50个全新场景进行三组150次运行：规则47/50、原单候选MPC48/50、简化版45/50，平均封顶帧数1000.96、908.50、974.02。
原MPC相对规则救回1场、退步0场，简化版救回0场、退步2场。全部完成并通过核验，模型已关闭；新结果与六场小测分开保存，不自动扩大或调参重跑。
2026-10-22按用户要求移除简化分支及专用入口，仅保留原MPC的一/两个LLM候选机制；简化版运行入口及移除前代码已归档至项目旁的`shepherding_sim_archive/retired_mpc_simplification_20261022/`。复现历史批次须使用其完整冻结源码，不能用当前代码替代。
随后新种子30001–30100的规则/原单候选MPC共200次运行完成并通过核验：91/100与92/100，平均封顶帧数1103.18与1094.79。MPC救回6场、退步5场，与先前新50场分开报告；模型已关闭，不自动追加测试。

## LLM选择修复模式的结论（2026-10-26）

之后把MPC评分改为**任务完成优先**：两个后台分支都预测到成功或全局2000帧预算，先比是否成功，再比完成帧。以上历史小测的结论不适用于此版本。
各批次只在批次内比较，不合并；失败按2000帧计入平均封顶帧数。

| 批次（10Hz，最多六场并行） | 结果 |
| --- | --- |
| 新200场探索（50001–50200） | 规则186、旧MPC185、任务完成优先MPC193；救回11、退步4，p≈0.12 |
| 独立400场确认（60001–60400） | 规则362/400 → LLM＋MPC 378/400；救回24、退步8，精确McNemar p≈0.0070 |
| 五组100场拆分（70001–70100） | 规则92、启发式直接95、LLM直接90、**启发式＋MPC 99**、LLM＋MPC 97；四项预注册比较Holm校正后均不显著 |

- 整个MPC组合提高了成功率；收益更像来自MPC筛选，而不是LLM候选。
- 事后复盘：LLM有少数局部提速（如70073只改模式提前268帧），但没有整体优势。
- 95次实际接管中，从实际接管状态重算有19次变为规则更好：后台预测与实际采用之间存在时机错配，未修复。
- 解释：异常类型已知并提供给控制器、修复模式只有三个、MPC内部模型与真实世界一致，简单启发式＋预测验证已经足够，LLM缺少发挥空间。
- 默认控制器和`config.py`没有因此切换；`config.py`和[SPARK_SETUP.md](SPARK_SETUP.md)中仍有Qwen3.8/8000端口的历史默认值，实验入口用冻结协议显式覆盖。

证据目录：`logs/experiments/mpc_completion_new200_10hz6_20261023/`、`mpc_confirmation_new400_10hz6_20261023/`、`mpc_component_new100_10hz6_20261024/`。
后续方向（预想外异常、去掉类型信息、LLM先诊断再选修复）尚待与指导教员确认。

本次整理保留过期、取消、输出合法性和模型身份检查；旧实验文件保持原样。
源代码整理后的版本与历史冻结版本不同，复现旧批次时需使用原源码快照。

## License

See [LICENSE](LICENSE).
