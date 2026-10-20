# Dual-Layer Autonomous Swarm Shepherding

异质羊群牧羊仿真：规则层逐帧控制狗，LLM在异常事件触发时提出模式或目标建议。
当前主线模型为本地优化 **Qwen3.6 NVFP4＋MTP3**。MPC候选比较尚未接入。

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

下一步讨论方案：双警报触发时比较 **规则方案＋两个LLM方案**，同一份高速模型并发生成两个候选。
先核验最新状态下的预测与实际执行一致性、候选重复和过期处理，再做小规模对照。
MPC尚未实现；预测窗口及双请求完整延迟仍待验证，不自动启动。

本次整理保留过期、取消、输出合法性和模型身份检查；旧实验文件保持原样。
源代码整理后的版本与历史冻结版本不同，复现旧批次时需使用原源码快照。

## License

See [LICENSE](LICENSE).
