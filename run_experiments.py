import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import config
from detector import Detector
from environment import SwarmEnv
from plot_analysis import plot_metrics_from_files
from high_level_controller import build_controller


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
    "GOAL_RETENTION_ENABLED",
    "GOAL_RETENTION_INNER_FRACTION",
    "GOAL_RETENTION_WEIGHT",
    "GOAL_RETENTION_VELOCITY_DAMPING",
    "HARD_TO_GUIDE_DISTANCE",
    "COHESION_REPAIR_DISTANCE",
    "CONTROLLER_KIND",
    "MOCK_LLM_LATENCY_FRAMES",
    "HIGH_LEVEL_HOLD_FRAMES",
    "HIGH_LEVEL_MIN_HOLD_FRAMES",
    "HIGH_LEVEL_MAX_HOLD_FRAMES",
    "HIGH_LEVEL_MIN_DWELL_FRAMES",
    "HIGH_LEVEL_MAX_RESPONSE_AGE_FRAMES",
    "HIGH_LEVEL_RETRY_FRAMES",
    "HIGH_LEVEL_CONTEXT_FRAMES",
    "HIGH_LEVEL_FORECAST_FRAMES",
    "REPAIR_TARGET_SELECTION_ENABLED",
    "COHESION_TARGET_MIN_HOLD_FRAMES",
    "HARD_TO_GUIDE_TARGET_MIN_HOLD_FRAMES",
    "REPAIR_TARGET_SWITCH_MARGIN",
    "REPAIR_TARGET_LOOKAHEAD_FRAMES",
    "REPAIR_TARGET_OUTSIDE_GOAL_FIRST",
    "LLM_API_TIMEOUT_SECONDS",
    "LLM_MAX_OUTPUT_TOKENS",
    "LLM_PENDING_POLL_SECONDS",
    "LLM_CACHE_ENABLED",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_CACHE_PATH",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_CACHE_PATH",
    "DEEPSEEK_REASONING_EFFORT",
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


def run_case(
    case,
    seed,
    max_steps,
    max_log_files,
    save_visuals=False,
    save_video=False,
    save_analysis=False,
    controller_kind="rule",
    llm_cache_enabled=True,
):
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
        LLM_CACHE_ENABLED=llm_cache_enabled,
    ):
        env = SwarmEnv(controller=build_controller(controller_kind))
        detector = make_detector()
        reached = False
        final_step = max_steps

        for step in range(max_steps):
            alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
            env.record_frame_metrics(metrics)
            reached = env.step(alarms)
            final_step = step + 1
            if (
                controller_kind in ("openai", "deepseek")
                and getattr(env.controller, "pending_request", None) is not None
            ):
                time.sleep(config.LLM_PENDING_POLL_SECONDS)
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

        row = summarize_case(
            case,
            seed,
            timestamp,
            data_path,
            control_path,
            video_path,
            trajectory_path,
            reached,
            final_step,
            env.goal_pos,
        )
        row["Controller"] = controller_kind
        env.close()
        return row


def summarize_case(
    case,
    seed,
    timestamp,
    data_path,
    control_path,
    video_path,
    trajectory_path,
    reached,
    final_step,
    goal_position,
):
    df = pd.read_parquet(data_path)
    metrics = (
        df[["Frame", "dist_to_goal", "dispersion", "mean_spread", "is_danger"]]
        .drop_duplicates()
        .sort_values("Frame")
        .reset_index(drop=True)
    )
    final = metrics.iloc[-1]
    sheep = df.loc[df["Agent_Type"] == "Sheep"].sort_values(
        ["Agent_ID", "Frame"]
    ).copy()
    sheep["inside_goal"] = (
        np.linalg.norm(
            sheep[["X", "Y"]].to_numpy() - np.asarray(goal_position),
            axis=1,
        )
        < config.GOAL_RADIUS
    )
    inside_counts = sheep.groupby("Frame")["inside_goal"].sum()
    goal_exits = sum(
        int(
            (
                group["inside_goal"].shift(fill_value=False)
                & ~group["inside_goal"]
            ).sum()
        )
        for _, group in sheep.groupby("Agent_ID")
    )

    control = pd.read_parquet(control_path)
    mode_counts = {key: int(value) for key, value in control["current_mode"].value_counts().items()}
    alarm_counts = {key: int(value) for key, value in control["alarms"].value_counts().items()}
    source_counts = {
        key: int(value)
        for key, value in control["decision_source"].value_counts().items()
    }
    high_status_counts = {
        key: int(value)
        for key, value in control["high_level_status"].dropna().value_counts().items()
    }
    request_ids = control["llm_request_id"].dropna().unique()
    accepted_responses = control.loc[
        (control["decision_source"] != "rule")
        & control["llm_request_id"].notna()
    ].drop_duplicates("llm_request_id")
    rejected_responses = control.loc[
        (control["high_level_status"] == "rejected")
        & control["llm_request_id"].notna()
    ].drop_duplicates("llm_request_id")
    returned_responses = pd.concat(
        [accepted_responses, rejected_responses], ignore_index=True
    ).drop_duplicates("llm_request_id")
    evaluated_selections = accepted_responses.loc[
        accepted_responses["llm_reference_mode"].notna()
    ]
    latency_frames = returned_responses["llm_latency_frames"].dropna()
    latency_seconds = returned_responses["llm_latency_seconds"].dropna()
    input_tokens = returned_responses["llm_input_tokens"].dropna()
    output_tokens = returned_responses["llm_output_tokens"].dropna()
    hold_frames = accepted_responses["llm_hold_frames"].dropna()
    cache_hits = returned_responses.loc[
        returned_responses["llm_cache_hit"] == True, "llm_request_id"
    ].dropna().unique()

    return {
        "Case": case["case"],
        "Seed": seed,
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
        "PeakInsideGoal": int(inside_counts.max()),
        "FinalInsideGoal": int(inside_counts.iloc[-1]),
        "GoalExits": goal_exits,
        "ModeCounts": mode_counts,
        "AlarmCounts": alarm_counts,
        "DecisionSourceCounts": source_counts,
        "HighLevelStatusCounts": high_status_counts,
        "HighLevelRequests": len(request_ids),
        "HighLevelResponses": len(returned_responses),
        "HighLevelAccepted": len(accepted_responses),
        "HighLevelRejected": len(rejected_responses),
        "HighLevelActiveFrames": int((control["decision_source"] != "rule").sum()),
        "MeanHighLevelHoldFrames": (
            round(float(hold_frames.mean()), 2) if not hold_frames.empty else ""
        ),
        "MeanHighLevelLatencyFrames": (
            round(float(latency_frames.mean()), 2) if not latency_frames.empty else ""
        ),
        "MeanHighLevelLatencySeconds": (
            round(float(latency_seconds.mean()), 4)
            if not latency_seconds.empty
            else ""
        ),
        "HighLevelCacheHits": len(cache_hits),
        "HighLevelSelectionsEvaluated": len(evaluated_selections),
        "HighLevelSelectionsCorrect": int(
            evaluated_selections["llm_selection_correct"].fillna(False).sum()
        ),
        "HighLevelSelectionAccuracy": (
            round(
                float(
                    evaluated_selections["llm_selection_correct"]
                    .fillna(False)
                    .mean()
                ),
                4,
            )
            if not evaluated_selections.empty
            else ""
        ),
        "HighLevelInputTokens": (
            int(input_tokens.sum())
            if not input_tokens.empty
            else 0
        ),
        "HighLevelOutputTokens": (
            int(output_tokens.sum())
            if not output_tokens.empty
            else 0
        ),
        "Data": str(data_path),
        "Control": str(control_path),
        "Video": str(video_path) if video_path.exists() else "",
        "Trajectory": str(trajectory_path) if trajectory_path.exists() else "",
    }


def write_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Case",
        "Seed",
        "Timestamp",
        "Purpose",
        "A",
        "B",
        "Repair",
        "Controller",
        "Success",
        "Frames",
        "FinalDist",
        "MinDist",
        "FinalDisp",
        "MaxDisp",
        "DangerFrames",
        "PeakInsideGoal",
        "FinalInsideGoal",
        "GoalExits",
        "ModeCounts",
        "AlarmCounts",
        "DecisionSourceCounts",
        "HighLevelStatusCounts",
        "HighLevelRequests",
        "HighLevelResponses",
        "HighLevelAccepted",
        "HighLevelRejected",
        "HighLevelActiveFrames",
        "MeanHighLevelHoldFrames",
        "MeanHighLevelLatencyFrames",
        "MeanHighLevelLatencySeconds",
        "HighLevelCacheHits",
        "HighLevelSelectionsEvaluated",
        "HighLevelSelectionsCorrect",
        "HighLevelSelectionAccuracy",
        "HighLevelInputTokens",
        "HighLevelOutputTokens",
        "Data",
        "Control",
        "Video",
        "Trajectory",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_metadata(output_path, args, seeds, selected_cases):
    source_names = (
        "config.py",
        "detector.py",
        "environment.py",
        "dynamics.py",
        "control_types.py",
        "low_level_controller.py",
        "mode_forecast.py",
        "high_level_controller.py",
        "responses_api_backend.py",
        "run_experiments.py",
    )
    metadata = {
        "created_at": datetime.now().isoformat(),
        "controller": args.controller,
        "seeds": seeds,
        "cases": [case["case"] for case in selected_cases],
        "max_steps": args.max_steps,
        "llm_cache_enabled": not args.no_llm_cache,
        "config": {
            key: value
            for key, value in vars(config).items()
            if key.isupper() and isinstance(value, (str, int, float, bool))
        },
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in source_names
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed C00-C08 demo experiment suite.")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="Run one fixed seed.")
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run several fixed seeds, for example: --seeds 21 22 23.",
    )
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation frames per case.")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case["case"] for case in CASES),
        help="Run only the listed cases, for example: --cases C08.",
    )
    parser.add_argument("--visuals", action="store_true", help="Save trajectory PNG files.")
    parser.add_argument("--video", action="store_true", help="Save MP4 animations. This is time-consuming and requires ffmpeg.")
    parser.add_argument("--analysis", action="store_true", help="Generate per-run analysis charts.")
    parser.add_argument("--controller", choices=("rule", "mock-llm", "openai", "deepseek"), default="rule", help="Controller architecture for every case.")
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Bypass saved API responses to measure a fresh provider call.",
    )
    parser.add_argument("--max-log-files", type=int, default=200, help="Temporary log retention limit during this run.")
    parser.add_argument(
        "--output",
        default=str(config.PROJECT_ROOT / "logs" / "experiment_summary.csv"),
        help="CSV path for the generated summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    seeds = list(dict.fromkeys(args.seeds or [args.seed if args.seed is not None else 70]))
    selected_cases = (
        [case for case in CASES if case["case"] in args.cases]
        if args.cases
        else CASES
    )

    if args.controller in ("openai", "deepseek"):
        try:
            probe = build_controller(args.controller)
        except RuntimeError as exc:
            raise SystemExit(f"Controller setup failed: {exc}") from None
        probe.close()

    print(
        f"Running {len(selected_cases)} cases x {len(seeds)} seeds "
        f"with max_steps={args.max_steps}"
    )
    for seed in seeds:
        for case in selected_cases:
            print(
                f"seed={seed} {case['case']}: {case['purpose']} "
                f"(A={case['num_abnormal_a']}, B={case['num_abnormal_b']}, "
                f"repair={case['repair_enabled']})"
            )
            row = run_case(
                case,
                seed=seed,
                max_steps=args.max_steps,
                max_log_files=args.max_log_files,
                save_visuals=args.visuals,
                save_video=args.video,
                save_analysis=args.analysis,
                controller_kind=args.controller,
                llm_cache_enabled=not args.no_llm_cache,
            )
            rows.append(row)
            print(
                f"  -> success={row['Success']}, frames={row['Frames']}, "
                f"final_dist={row['FinalDist']}, "
                f"danger_frames={row['DangerFrames']}, ts={row['Timestamp']}"
            )

    output_path = Path(args.output)
    write_summary(rows, output_path)
    metadata_path = write_experiment_metadata(
        output_path, args, seeds, selected_cases
    )
    print(f"Summary saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
