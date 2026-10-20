# Spark模型服务

`qwen36_fast_sync.py` 是已部署高速模型启停脚本的本地源码副本。
它在Spark的Linux环境运行，不是在Windows上直接执行的仿真入口。

- NVIDIA Sync入口：**Qwen3.6 NVFP4 + MTP3**。
- 远端脚本：`~/.local/bin/shepherding-qwen36-fast-sync.py`。
- 模型服务端口：`8002`。
- 停止时只停止本次拥有的高速服务，不启动FP8。

`benchmark_session.py` 为性能测试管理同一份高速模型的使用期：测试前关闭则由测试启动、结束后停止；原本运行则保留。它核验原启动参数，不启动FP8或第二份模型，也不修改Sync按钮。

部署和历史配置说明见[SPARK_SETUP.md](../../SPARK_SETUP.md)。
