# Swarm Shepherding Simulation

A Multi-Agent Simulation Framework for CSS Laboratory, Hiroshima University.

---

## 概要 (Overview)

**[JP]**
本プロジェクトは、広島大学CSS研究室における研究活動を支援するために開発された、マルチエージェントベースのシミュレーション環境です。本シミュレータは、群知能モデルである **Boids** と、**Strömbom** の牧羊アルゴリズムを実装しており、牧羊犬による羊群の制御ダイナミクスを再現します。主要な機能として、羊群の凝集度に応じて牧羊犬の行動戦略を「収集（Collect）」と「駆動（Drive）」の間で切り替える、状態遷移ベースの制御ロジックが組み込まれています。

**[EN]**
This project is a multi-agent-based simulation environment developed to support research activities at the CSS Laboratory, Hiroshima University. The simulator implements the **Boids** swarm intelligence model and **Strömbom's** shepherding algorithm to reproduce the dynamics of flock control by a shepherd dog. A key feature is its state-transition-based control logic, which enables the dog to switch its strategy between "collecting" dispersed sheep and "driving" the cohesive flock based on the group's cohesion level.

---

## 主な機能 (Features)

- **再現性のあるシミュレーション (Reproducible Simulations)**
  - 固定ランダムシード (`config.USE_FIXED_SEED`) によりエージェントの初期位置を制御し、実験の再現性を確保します。
  - Ensures consistent experimental conditions through a fixed random seed, which governs the initial positions of agents.

- **Strömbomベースの状態遷移 (Strömbom-based State Machine)**
  - 牧羊犬の行動は2つのモードを動的に切り替えます。
  - The shepherd dog's behavior dynamically switches between two modes:
    - **収集モード (Collect Mode)**: 羊群の分散度がしきい値 (`COHESION_THRESHOLD`) を超えると起動。最も遠い羊を群れの中心に戻すように動きます。
    - Activated when the flock's dispersion exceeds a predefined threshold. The dog targets the furthest sheep to herd it back towards the flock's center of mass.
    - **駆動モード (Drive Mode)**: 羊群が十分に凝集している場合に起動。群れ全体の背後に位置し、目標地点へ向かって押し進めます。
    - Activated when the flock is cohesive. The dog positions itself behind the flock's center of mass to push it towards the goal.

- **物理ベースの相互作用 (Physics-based Agent Interaction)**
  - **距離減衰する反発力 (Distance-Decaying Repulsion)**: 犬から羊への反発力は距離に反比例し、より現実的な回避行動をモデル化します。
  - The repulsion force exerted by the dog on the sheep is inversely proportional to the distance, creating a more realistic avoidance behavior.

- **異常状態の敵対的テスト (Abnormal Status Adversarial Testing)**
  - 危険判定アルゴリズムのストレステスト用に、2種類の異常個体を生成できます：
  - Supports generating two types of abnormal agents for stress-testing:
    - **Type A (無反応 / Unresponsive)**: 牧羊犬からの反発力と群れの凝集力を完全に無視します。 / Completely ignores dog repulsion and flock cohesion.
    - **Type B (分散 / Dispersing)**: 凝集力が極端に低く、他個体への分離力が異常に高いため、群れの境界をさまよいます。 / Exhibits extremely low cohesion and highly amplified separation force, wandering at the edges.

- **高性能データロギング (High-Performance Data Logging)**
  - 各フレームのエージェント状態（位置、ID）を、大規模データ分析に適した **Apache Parquet** 形式で記録します。
  - Agent states for each frame are recorded in the efficient Apache Parquet format, ideal for large-scale data analysis.

- **データと同期した可視化出力 (Synchronized Visual & Data Outputs)**
  - シミュレーション完了時、データファイルとタイムスタンプが一致する静的な軌跡プロット（PNG）とアニメーション（GIF）を生成します。
  - At the end of each simulation, a static trajectory plot (PNG) and an animated GIF are generated with timestamps that match the corresponding data file.

- **ログの自動ローテーション (Automated Log Rotation)**
  - ログファイル数 (`MAX_LOG_FILES`) を一定に保ち、最も古いデータセット（Parquet, PNG, GIF）を自動的に削除してディスクスペースを管理します。
  - A built-in cleanup mechanism maintains a fixed number of recent logs and automatically deletes the oldest data sets to manage disk space.

---

## ディレクトリ構成 (Directory Structure)
```text
shepherding_sim/
├── main.py              # シミュレーション実行のメインスクリプト (Main simulation entry point)
├── environment.py       # 中核となるシミュレーション環境、物理演算、エージェントロジック (Core simulation environment, physics, and agent logic)
├── config.py            # 全ての調整可能なパラメータとシミュレーション設定 (All tunable parameters and simulation settings)
├── visualizer.py        # リアルタイム描画と可視化出力（PNG/GIF）の生成 (Real-time rendering and visual output generation)
└── logs/                # 全ての出力ファイルが保存される自動生成ディレクトリ (Auto-generated directory for all outputs)
    ├── data/            # シミュレーションデータを *.parquet 形式で格納 (Stores simulation data in *.parquet format)
    └── visuals/         # 軌跡画像 trajectory_*.png とアニメーション animation_*.gif を格納 (Stores trajectory and animation files)
```

---

## ⚙️ パラメータ設定 (Configuration)
シミュレーションの挙動は `config.py` で調整可能です。
The simulation behavior can be tuned in `config.py`.

| Parameter (変数) | Value | Description (説明) |
| :--- | :---: | :--- |
| `SHEEP_MAX_SPEED` | 0.7 | 羊の最大速度。質量感（重み）を持たせるために低めに設定。<br>Max speed of sheep. Kept low to simulate mass. |
| `DOG_MAX_SPEED` | 1.5 | 犬の最大速度。羊を回り込むための十分な機動力。<br>Max speed of dog. Faster than sheep for maneuvering. |
| `WEIGHT_COHESION` | 0.01 | 羊が群れ（中心）へまとまろうとする力。<br>Flock's desire to converge to the center. |
| `WEIGHT_SEPARATION` | 2.0 | 羊同士の重なりを防ぐ反発力（距離3.0以内で作動）。<br>Force preventing sheep from overlapping (activates < 3.0). |
| `WEIGHT_DOG_REPULSION` | 10.0 | 犬への恐怖（犬から逃げる強さ）。距離に反比例。<br>Fear of the dog. Inversely proportional to distance. |
| `DOG_SENSING_RANGE` | 50.0 | 羊が犬を感知して逃げ始める半径距離。<br>Radius within which sheep react to the dog. |
| `COHESION_THRESHOLD` | 25.0 | 犬が「Drive」から「Collect」に移行する羊の散らばりしきい値。<br>Max distance to COM that triggers "Collect" mode. |
| `MAX_LOG_FILES` | 30 | 保持するログファイルの最大数（同期ローテーション）。<br>Maximum number of synchronized log files to keep. |
| `NUM_ABNORMAL_A` | 2 | Type A（無反応 / Unresponsive）の異常羊の数。環境を切り替えるにはこれを変更します。<br>Number of Type A abnormal sheep. Modify this to toggle environments. |
| `NUM_ABNORMAL_B` | 2 | Type B（分散 / Dispersing）の異常羊の数。環境を切り替えるにはこれを変更します。<br>Number of Type B abnormal sheep. Modify this to toggle environments. |

---

## 🚀 実行方法 (How to Run)

**1. 依存ライブラリのインストール (Install Dependencies)**
```bash
pip install numpy pandas pyarrow matplotlib
```
*※ GIFを保存する場合は、環境により `Pillow` または `imagemagick` が必要になる場合があります。*
*(Note: `Pillow` or `imagemagick` may be required for saving GIF animations.)*

**2. シミュレーションの実行 (Run Simulation)**
```bash
python main.py
```
実行が完了すると、自動的に `logs/data` および `logs/visuals` フォルダが作成され、結果が保存されます。
Upon completion, the `logs/data` and `logs/visuals` folders will be automatically created and populated with results.

> **💡 Tip (便利なヒント):**
> VSCodeなどのエディタに **`parquet-viewer`** 拡張機能をインストールすると、保存された `.parquet` ファイルの中身を手軽にプレビューすることができます。
> You can install an extension like **`parquet-viewer`** in your IDE (e.g., VSCode) to easily browse and preview the saved `.parquet` data files.


---

## 開発者・共同研究者の方へ (For Developers & Collaborators)

本リポジトリでは、シミュレーションによって生成される大容量のデータファイルや画像ファイル（`.parquet`, `.png`, `.gif`）は、Git の追跡対象外（`.gitignore`）に設定されています。新しくリポジトリをクローンして実行した場合、`logs/` ディレクトリはローカル環境にのみ自動生成され、リモートリポジトリ（GitHubなど）の容量を圧迫することはありません。

In this repository, the generated simulation data and visual files (`.parquet`, `.png`, `.gif`) are explicitly ignored by Git via `.gitignore`. When you clone the repository and run the simulation, the `logs/` directory will be automatically generated locally, preventing these large files from bloating the remote repository.

---

##  データ仕様 (Data Schema) - [学生2へ / For Student 2]

学生2の「危険判定アルゴリズム (Danger Judgment Algorithm)」開発用データスキーマです。
シミュレーション結果は `logs/data/data_YYYYMMDD_HHMMSS.parquet` として出力されます。

This section details the Parquet data schema for Student 2's downstream "Danger Judgment Algorithm" development.

###  Schema Definition

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| **Frame** | `int` | 現在のフレーム番号 (Time step) | `125` |
| **Agent_Type**| `str` | エージェントの種類 (`Dog` または `Sheep`) | `'Sheep'` |
| **Agent_ID** | `int` | エージェント固有のID (Dog=0, Sheep=0~19) | `5` |
| **X** | `float` | X座標 (小数点2桁丸め) | `120.45` |
| **Y** | `float` | Y座標 (小数点2桁丸め) | `85.32` |
| **Status** | `str` | エージェントの真の状態・異常ラベル (`Normal`, `A`, `B`, `Dog`)。<br>True identity/status of the agent. | `'A'` |

###  Pandas 読み込みサンプル (Python Read Example)
以下は、エクスポートされた Parquet ファイルを読み込み、特定の羊の軌跡や犬の動きを抽出するサンプルコードです。

```python
import pandas as pd

# Parquet ファイルの読み込み
df = pd.read_parquet("logs/data/data_20231024_153000.parquet")

# 1. 犬 (Dog) のデータだけを抽出
dog_data = df[df['Agent_Type'] == 'Dog']
print("Dog Trajectory:\n", dog_data.head())

# 2. 特定のフレーム (例: Frame 100) の全エージェントの座標を取得
frame_100 = df[df['Frame'] == 100]

# 3. 羊たちの重心 (Center of Mass) を計算する例
sheep_data = df[df['Agent_Type'] == 'Sheep']
com_per_frame = sheep_data.groupby('Frame')[['X', 'Y']].mean()
print("\nCenter of Mass per Frame:\n", com_per_frame.head())
```

アルゴリズムの開発、応援しております！何か不明点があればお気軽にお知らせください。
Good luck with the algorithm development!

---

## 🤝 開発者連携ガイド (Developer Integration Guide)

本プロジェクトのダウンストリーム開発を担当する学生2と学生3のための統合ガイドです。
This is the integration guide for Student 2 and Student 3 handling the downstream development.

###  学生2へ: 危険判定 (Anomaly Detection) 開発ガイド

- **データ取得とアノテーション (Data Acquisition & Ground Truth)**:
  エクスポートされる `.parquet` ログファイルには、すでに `Status` 列 (`'Normal'`, `'A'`, `'B'`) が含まれています。アルゴリズムの学習および評価フェーズにおいて、この列を**正解ラベル (Ground Truth)** として利用し、検出アルゴリズムの精度を検証してください。
  *The exported `.parquet` log files already contain the `Status` column. Use this column as the Ground Truth during the training and evaluation phases of your algorithm.*
- **環境コントロール (Environment Control)**:
  開発・テストフェーズにおいて、異なる異常個体比率のデータセットを生成したい場合は、`config.py` 内の `NUM_ABNORMAL_A` および `NUM_ABNORMAL_B` の数値を変更してシミュレーションを実行してください。
  *To generate datasets with different anomaly ratios, modify `NUM_ABNORMAL_A` and `NUM_ABNORMAL_B` in `config.py`.*

###  学生3へ: 修復モード (Repair Mode) 接入ガイド

- **API 詳細 (API Details)**:
  `environment.py` 内の `SwarmEnv` クラスには、介入用 API として `repair_sheep(indices)` が実装されています。このメソッドは非常に柔軟に設計されており、対象となる羊の単一の整数 ID、ID のリスト、または NumPy 配列のいずれの形式でも引数として受け付けることができます。
  *The `repair_sheep(indices)` API is extremely flexible and accepts a single integer ID, a list of IDs, or a NumPy array.*

- **メインループ統合サンプル (Integration Example)**:
  将来的に、お二人のアルゴリズムを `main.py` のシミュレーションループに統合する際の擬似コード（Pseudo-code）は以下のようになります。

  ```python
  while step_count < config.MAX_STEPS:
      reached_goal = env.step()
      
      # --- [Student 2] Anomaly Detection ---
      # 現在の物理状態 (env.sheep_pos, env.sheep_vel) から異常個体のIDを予測
      predicted_abnormal_ids = student2_detect_anomaly(env.sheep_pos, env.sheep_vel)
      
      # --- [Student 3] Repair Intervention ---
      # 検出されたIDをAPIに渡し、対象の羊を物理的に「正常」状態へ強制修復
      if len(predicted_abnormal_ids) > 0:
          env.repair_sheep(predicted_abnormal_ids)
          
      vis.render(env)
      # ...
  ```