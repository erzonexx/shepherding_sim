import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from detector import Detector
from environment import SwarmEnv
from plot_analysis import plot_metrics_from_files


CASES = [
    {
        "case": "C00",
        "purpose": "Normal baseline",
        "num_abnormal_a": 0,
        "num_abnormal_b": 0,
        "repair_enabled": False,
    },
    {
        "case": "C01",
        "purpose": "Mixed anomaly, no repair",
        "num_abnormal_a": 2,
        "num_abnormal_b": 2,
        "repair_enabled": False,
    },
    {
        "case": "C02",
        "purpose": "Mixed anomaly, repair on",
        "num_abnormal_a": 2,
        "num_abnormal_b": 2,
        "repair_enabled": True,
    },
    {
        "case": "C03",
        "purpose": "A-only, no repair",
        "num_abnormal_a": 4,
        "num_abnormal_b": 0,
        "repair_enabled": False,
    },
    {
        "case": "C04",
        "purpose": "A-only, repair on",
        "num_abnormal_a": 4,
        "num_abnormal_b": 0,
        "repair_enabled": True,
    },
    {
        "case": "C05",
        "purpose": "B-only, no repair",
        "num_abnormal_a": 0,
        "num_abnormal_b": 4,
        "repair_enabled": False,
    },
    {
        "case": "C06",
        "purpose": "B-only, repair on",
        "num_abnormal_a": 0,
        "num_abnormal_b": 4,
        "repair_enabled": True,
    },
    {
        "case": "C07",
        "purpose": "Stress mixed, no repair",
        "num_abnormal_a": 3,
        "num_abnormal_b": 3,
        "repair_enabled": False,
    },
    {
        "case": "C08",
        "purpose": "Stress mixed, repair on",
        "num_abnormal_a": 3,
        "num_abnormal_b": 3,
        "repair_enabled": True,
    },
]


CONFIG_KEYS = [
    "USE_FIXED_SEED",
    "RANDOM_SEED",
    "MAX_STEPS",
    "NUM_ABNORMAL_A",
    "NUM_ABNORMAL_B",
    "REPAIR_ENABLED",
    "SAVE_TRAJECTORY_PNG",
    "SAVE_ANIMATION_MP4",
    "MAX_LOG_FILES",
]


@contextmanager
def temporary_config(**overrides):
    original = {key: getattr(config, key) for key in CONFIG_KEYS}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def make_detector():
    return Detector(
        dispersion_threshold=config.DANGER_DISPERSION_THRESHOLD,
        mean_spread_threshold=config.DANGER_MEAN_SPREAD_THRESHOLD,
        stagnation_frames=config.DANGER_STAGNATION_FRAMES,
        stagnation_threshold=config.DANGER_STAGNATION_THRESHOLD,
    )


def run_case(case, seed, max_steps, max_log_files, save_visuals=False, save_video=False, save_analysis=False):
    with temporary_config(
        USE_FIXED_SEED=True,
        RANDOM_SEED=seed,
        MAX_STEPS=max_steps,
        NUM_ABNORMAL_A=case["num_abnormal_a"],
        NUM_ABNORMAL_B=case["num_abnormal_b"],
        REPAIR_ENABLED=case["repair_enabled"],
        SAVE_TRAJECTORY_PNG=save_visuals,
        SAVE_ANIMATION_MP4=save_video,
        MAX_LOG_FILES=max_log_files,
    ):
        env = SwarmEnv()
        detector = make_detector()
        reached = False
        final_step = max_steps

        for step in range(max_steps):
            alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
            env.record_frame_metrics(metrics)
            reached = env.step(alarms)
            final_step = step + 1
            if reached:
                break

        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{case['case']}"
        env.save_data_to_parquet(timestamp)
        env.save_control_to_parquet(timestamp)

        if save_visuals or save_video:
            from matplotlib import pyplot as plt
            from visualizer import Visualizer

            vis = Visualizer(env)
            if save_visuals:
                vis.save_trajectory(env, timestamp)
            if save_video:
                vis.save_animation_mp4(env, timestamp)
            plt.close(vis.fig)

        data_path = Path(config.DATA_LOG_DIR) / f"data_{timestamp}.parquet"
        control_path = Path(config.CONTROL_LOG_DIR) / f"control_{timestamp}.parquet"
        video_path = Path(config.VIDEO_LOG_DIR) / f"animation_{timestamp}.mp4"
        trajectory_path = Path(config.VISUAL_LOG_DIR) / f"trajectory_{timestamp}.png"

        if save_analysis:
            plot_metrics_from_files([str(data_path)], config.ANALYSIS_LOG_DIR)

        return summarize_case(case, timestamp, data_path, control_path, video_path, trajectory_path, reached, final_step)


def summarize_case(case, timestamp, data_path, control_path, video_path, trajectory_path, reached, final_step):
    df = pd.read_parquet(data_path)
    metrics = (
        df[["Frame", "dist_to_goal", "dispersion", "mean_spread", "is_danger"]]
        .drop_duplicates()
        .sort_values("Frame")
        .reset_index(drop=True)
    )
    final = metrics.iloc[-1]

    control = pd.read_parquet(control_path)
    mode_counts = {key: int(value) for key, value in control["current_mode"].value_counts().items()}
    alarm_counts = {key: int(value) for key, value in control["alarms"].value_counts().items()}

    return {
        "Case": case["case"],
        "Timestamp": timestamp,
        "Purpose": case["purpose"],
        "A": case["num_abnormal_a"],
        "B": case["num_abnormal_b"],
        "Repair": case["repair_enabled"],
        "Success": reached,
        "Frames": final_step,
        "FinalDist": round(float(final["dist_to_goal"]), 2),
        "MinDist": round(float(metrics["dist_to_goal"].min()), 2),
        "FinalDisp": round(float(final["dispersion"]), 2),
        "MaxDisp": round(float(metrics["dispersion"].max()), 2),
        "DangerFrames": int(metrics["is_danger"].sum()),
        "ModeCounts": mode_counts,
        "AlarmCounts": alarm_counts,
        "Data": str(data_path),
        "Control": str(control_path),
        "Video": str(video_path) if video_path.exists() else "",
        "Trajectory": str(trajectory_path) if trajectory_path.exists() else "",
    }


def write_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Case",
        "Timestamp",
        "Purpose",
        "A",
        "B",
        "Repair",
        "Success",
        "Frames",
        "FinalDist",
        "MinDist",
        "FinalDisp",
        "MaxDisp",
        "DangerFrames",
        "ModeCounts",
        "AlarmCounts",
        "Data",
        "Control",
        "Video",
        "Trajectory",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed C00-C08 demo experiment suite.")
    parser.add_argument("--seed", type=int, default=70, help="Fixed random seed for all cases.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation frames per case.")
    parser.add_argument("--visuals", action="store_true", help="Save trajectory PNG files.")
    parser.add_argument("--video", action="store_true", help="Save MP4 animations. This is slow and requires ffmpeg.")
    parser.add_argument("--analysis", action="store_true", help="Generate per-run analysis charts.")
    parser.add_argument("--max-log-files", type=int, default=200, help="Temporary log retention limit during this run.")
    parser.add_argument(
        "--output",
        default="project_docs/presentation/experiment_summary_new.csv",
        help="CSV path for the generated summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    print(f"Running {len(CASES)} demo cases with seed={args.seed}, max_steps={args.max_steps}")
    for case in CASES:
        print(
            f"{case['case']}: {case['purpose']} "
            f"(A={case['num_abnormal_a']}, B={case['num_abnormal_b']}, repair={case['repair_enabled']})"
        )
        row = run_case(
            case,
            seed=args.seed,
            max_steps=args.max_steps,
            max_log_files=args.max_log_files,
            save_visuals=args.visuals,
            save_video=args.video,
            save_analysis=args.analysis,
        )
        rows.append(row)
        print(
            f"  -> success={row['Success']}, frames={row['Frames']}, "
            f"final_dist={row['FinalDist']}, danger_frames={row['DangerFrames']}, ts={row['Timestamp']}"
        )

    output_path = Path(args.output)
    write_summary(rows, output_path)
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    main()
