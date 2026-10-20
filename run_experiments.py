import argparse
import csv
from contextlib import contextmanager
from datetime import datetime
import random
from pathlib import Path
import time

import numpy as np
import pandas as pd

import config
from detector import Detector
from environment import SwarmEnv
from plot_analysis import plot_metrics_from_files
from high_level_controller import build_controller
from low_level_controller import rule_mode_for_alarms
from experiment_audit import (SOURCE_NAMES, INPUT_SEMANTICS_VERSION, backend_snapshot,
                              config_snapshot, source_hashes, write_json_exclusive,
                              begin_run_record, finish_run_record)


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
    "NUM_ABNORMAL_C",
    "REPAIR_ENABLED",
    "SAVE_TRAJECTORY_PNG",
    "SAVE_ANIMATION_MP4",
    "MAX_LOG_FILES",
    "LOG_RETENTION_ENABLED",
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
    "HIGH_LEVEL_EXPLICIT_EXECUTION",
    "HIGH_LEVEL_TARGET_HANDOFF_ENABLED",
    "HIGH_LEVEL_FORECAST_FRAMES",
    "HIGH_LEVEL_CONCURRENT_MIN_FRAMES",
    "HIGH_LEVEL_ONE_REQUEST_PER_EPISODE",
    "HIGH_LEVEL_TARGET_SELECTION_ENABLED",
    "HIGH_LEVEL_MODE_SELECTION_ENABLED",
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
    "VLLM_MODEL",
    "VLLM_BASE_URL",
    "VLLM_CACHE_PATH",
    "VLLM_MAX_OUTPUT_TOKENS",
    "VLLM_COMPACT_OUTPUT",
    "VLLM_INPUT_FORMAT",
    "VLLM_PREFIX_PRIMING",
    "VLLM_PREFIX_TOKEN_FILE",
]


@contextmanager
def temporary_config(**overrides):
    original = {
        key: getattr(config, key) for key in set(CONFIG_KEYS) | set(overrides)
    }
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
        NUM_ABNORMAL_C=case.get("num_abnormal_c", 0),
        REPAIR_ENABLED=case["repair_enabled"],
        SAVE_TRAJECTORY_PNG=save_visuals,
        SAVE_ANIMATION_MP4=save_video,
        MAX_LOG_FILES=max_log_files,
        LLM_CACHE_ENABLED=llm_cache_enabled,
        CONTROLLER_KIND=controller_kind,
    ):
        env = SwarmEnv(controller=build_controller(controller_kind))
        try:
            timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{case['case']}"
            manifest_path = Path(config.CONTROL_LOG_DIR) / f"run_{timestamp}.json"
            manifest = begin_run_record(manifest_path, env.controller, config,
                                        case=case, seed=seed, controller_kind=controller_kind)
            run_started = time.perf_counter()
            detector = make_detector()
            reached = False
            final_step = max_steps

            for step in range(max_steps):
                alarms, metrics, _ = detector.detect(env.sheep_pos, env.goal_pos)
                env.record_frame_metrics(metrics)
                reached = env.step(alarms)
                final_step = step + 1
                if (
                    controller_kind in ("openai", "deepseek", "vllm")
                    and getattr(env.controller, "pending_request", None) is not None
                ):
                    time.sleep(config.LLM_PENDING_POLL_SECONDS)
                if reached:
                    break

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
            finish_run_record(manifest_path, manifest, success=reached, frames=final_step,
                              wall_seconds=time.perf_counter() - run_started,
                              data=str(data_path), control=str(control_path))
            row["RunMetadata"] = str(manifest_path)
            row["RequestsAudit"] = manifest.get("requests_audit", "")
            return row
        finally:
            env.close()


def count_high_level_override_frames(control, repair_enabled):
    active = control.loc[control["decision_source"] != "rule"]
    return sum(
        bool(getattr(row, "llm_target_override", False)) or row.current_mode
        != rule_mode_for_alarms(
            () if row.alarms == "none" else tuple(row.alarms.split(";")),
            repair_enabled,
        )
        for row in active.itertuples()
    )


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
    delegated_responses = control.loc[
        (control['high_level_status'] == 'delegated') & control['llm_request_id'].notna()
    ].drop_duplicates('llm_request_id')
    returned_responses = pd.concat(
        [accepted_responses, rejected_responses, delegated_responses], ignore_index=True
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
        "C": case.get("num_abnormal_c", 0),
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
        "HighLevelDelegated": len(delegated_responses),
        "HighLevelActiveFrames": int((control["decision_source"] != "rule").sum()),
        "HighLevelOverrideFrames": count_high_level_override_frames(
            control, case["repair_enabled"]
        ),
        "HighLevelTargetOverrideFrames": int(
            control.get("llm_target_override", pd.Series(False, index=control.index)).fillna(False).sum()
        ),
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
        "C",
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
        "HighLevelDelegated",
        "HighLevelActiveFrames",
        "HighLevelOverrideFrames",
        "HighLevelTargetOverrideFrames",
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
        "RunMetadata",
        "RequestsAudit",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_metadata(output_path, args, seeds, selected_cases, backend_metadata=None, source_dir=None):
    metadata = {
        "created_at": datetime.now().isoformat(),
        "controller": args.controller,
        "seeds": seeds,
        "cases": [case["case"] for case in selected_cases],
        "case_settings": selected_cases,
        "seed_plan": {
            "mode": args.seed_mode if args.runs is not None else "explicit",
            "batch_seed": args.batch_seed if args.runs is not None else None,
        },
        "max_steps": args.max_steps,
        "llm_cache_enabled": not args.no_llm_cache,
        "config": config_snapshot(config),
        "source_sha256": source_hashes(config.PROJECT_ROOT),
        "backend": backend_metadata,
        "input_semantics_version": INPUT_SEMANTICS_VERSION,
        "pacing": {"fixed_wall_time_step": False,
                   "pending_api_poll_seconds": config.LLM_PENDING_POLL_SECONDS},
    }
    if source_dir is not None:
        source_dir.mkdir(parents=True, exist_ok=False)
        for name in SOURCE_NAMES:
            (source_dir / name).write_bytes((config.PROJECT_ROOT / name).read_bytes())
        metadata["source_snapshot"] = str(source_dir)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_json_exclusive(metadata_path, metadata)
    return metadata_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run batch shepherding experiments.")
    parser.add_argument('--explicit-execution', action='store_true',
                        help='Experimental FOLLOW_RULE versus deliberate COMMIT, full JSON only.')
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="Run one fixed seed.")
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run several fixed seeds, for example: --seeds 21 22 23.",
    )
    seed_group.add_argument("--runs", type=int, help="Number of distinct seeds to generate per scenario.")
    parser.add_argument("--seed-mode", choices=("random", "sequence"), default="random")
    parser.add_argument(
        "--batch-seed", type=int, default=70,
        help="Random seed-group identifier, or the first seed in sequence mode.",
    )
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation frames per case.")
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case["case"] for case in CASES),
        help="Run only the listed cases, for example: --cases C08.",
    )
    scenario_group.add_argument("--scenario", choices=(*config.SCENARIOS, "custom"))
    parser.add_argument("--abnormal-a", type=int)
    parser.add_argument("--abnormal-b", type=int)
    parser.add_argument("--abnormal-c", type=int)
    parser.add_argument(
        "--repair", action=argparse.BooleanOptionalAction, default=None,
        help="Repair on/off for --scenario; existing --cases retain their definitions.",
    )
    parser.add_argument(
        "--export", choices=("data", "plots", "video"), default="data",
        help="Batch exports: data only, data plus plots, or data plus plots and MP4.",
    )
    parser.add_argument("--visuals", action="store_true", help="Save trajectory PNG files.")
    parser.add_argument("--video", action="store_true", help="Save MP4 animations. This is time-consuming and requires ffmpeg.")
    parser.add_argument("--analysis", action="store_true", help="Generate per-run analysis charts.")
    parser.add_argument("--controller", choices=("rule", "mock-llm", "openai", "deepseek", "vllm"), default="rule", help="Controller architecture for every case.")
    permission_group = parser.add_mutually_exclusive_group()
    permission_group.add_argument(
        "--llm-mode-only", action="store_true",
        help="Ablation: LLM selects a mode but must delegate repair targets to rules.",
    )
    permission_group.add_argument(
        "--llm-target-only", action="store_true",
        help="Ablation: rules select the mode; LLM may select a repair target only.",
    )
    parser.add_argument(
        "--concurrent-min-frames", type=int,
        help="Require this many consecutive dual-alarm frames before an API request.",
    )
    parser.add_argument(
        "--one-request-per-concurrent-episode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Submit at most one API request until the dual-alarm episode ends.",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Bypass saved API responses to measure a fresh provider call.",
    )
    parser.add_argument("--max-log-files", type=int, default=200, help="Temporary log retention limit during this run.")
    parser.add_argument(
        "--output",
        default=None,
        help="Summary CSV path; default is a new timestamped logs/experiments folder.",
    )
    parser.add_argument(
        "--vllm-compact-output", action=argparse.BooleanOptionalAction, default=None,
        help="Use compact m/t/h JSON for vllm; --no-vllm-compact-output restores verbose JSON.",
    )
    parser.add_argument(
        "--vllm-prefix-priming", action=argparse.BooleanOptionalAction, default=None,
        help="Warm fixed instructions once for vllm; requires VLLM_PREFIX_TOKEN_FILE.",
    )
    parser.add_argument(
        "--vllm-input-format", choices=("json", "tables", "dedup"), default=None,
        help="Use original JSON, experimental tables or deduplicated named JSON for vllm.",
    )
    args = parser.parse_args(argv)
    if args.vllm_input_format is not None and args.controller != "vllm":
        parser.error("--vllm-input-format requires --controller vllm")
    if args.vllm_prefix_priming is not None and args.controller != "vllm":
        parser.error("--vllm-prefix-priming requires --controller vllm")
    if args.vllm_compact_output is not None and args.controller != "vllm":
        parser.error("--vllm-compact-output requires --controller vllm")
    if (args.llm_mode_only or args.llm_target_only) and args.controller not in ("openai", "deepseek", "vllm"):
        parser.error("LLM permission ablations require an API controller.")
    if args.max_steps < 1 or args.max_log_files < 1:
        parser.error("--max-steps and --max-log-files must be positive.")
    if args.concurrent_min_frames is not None and args.concurrent_min_frames < 1:
        parser.error("--concurrent-min-frames must be positive.")
    if args.runs is not None and not 1 <= args.runs <= 2**32:
        parser.error("--runs must be between 1 and 4294967296.")
    for seed in [args.batch_seed, *([] if args.seed is None else [args.seed]), *(args.seeds or [])]:
        if not 0 <= seed < 2**32:
            parser.error("Seeds must be between 0 and 4294967295.")
    if args.runs and args.seed_mode == "sequence" and args.batch_seed + args.runs > 2**32:
        parser.error("The seed sequence exceeds 4294967295.")
    if args.scenario is None and any(value is not None for value in (args.abnormal_a, args.abnormal_b, args.abnormal_c, args.repair)):
        parser.error("Population and repair overrides require --scenario.")
    if args.scenario:
        counts = config.SCENARIOS.get(args.scenario, (config.NUM_ABNORMAL_A, config.NUM_ABNORMAL_B, config.NUM_ABNORMAL_C))
        args.abnormal_a = counts[0] if args.abnormal_a is None else args.abnormal_a
        args.abnormal_b = counts[1] if args.abnormal_b is None else args.abnormal_b
        args.abnormal_c = counts[2] if args.abnormal_c is None else args.abnormal_c
        population = (args.abnormal_a, args.abnormal_b, args.abnormal_c)
        if min(population) < 0 or sum(population) > config.NUM_SHEEP:
            parser.error("A/B/C counts must be nonnegative and their sum cannot exceed NUM_SHEEP.")
        args.repair = config.REPAIR_ENABLED if args.repair is None else args.repair
    args.visuals = args.visuals or args.export in ("plots", "video")
    args.analysis = args.analysis or args.export in ("plots", "video")
    args.video = args.video or args.export == "video"
    return args


def select_seeds(args):
    if args.runs is not None:
        if args.seed_mode == "sequence":
            return list(range(args.batch_seed, args.batch_seed + args.runs))
        return random.Random(args.batch_seed).sample(range(2**32), args.runs)
    return list(dict.fromkeys(args.seeds or [args.seed if args.seed is not None else 70]))


def select_cases(args):
    if args.scenario:
        return [{
            "case": args.scenario,
            "purpose": f"{args.scenario} population, repair {'on' if args.repair else 'off'}",
            "num_abnormal_a": args.abnormal_a,
            "num_abnormal_b": args.abnormal_b,
            "num_abnormal_c": args.abnormal_c,
            "repair_enabled": args.repair,
        }]
    return (
        [case for case in CASES if case["case"] in args.cases]
        if args.cases
        else CASES
    )


def main(argv=None):
    args = parse_args(argv)
    overrides = {'HIGH_LEVEL_EXPLICIT_EXECUTION': args.explicit_execution}
    if args.vllm_input_format is not None:
        overrides["VLLM_INPUT_FORMAT"] = args.vllm_input_format
    if args.vllm_compact_output is not None:
        overrides["VLLM_COMPACT_OUTPUT"] = args.vllm_compact_output
    if args.vllm_prefix_priming is not None:
        overrides["VLLM_PREFIX_PRIMING"] = args.vllm_prefix_priming
    if args.llm_mode_only:
        overrides["HIGH_LEVEL_TARGET_SELECTION_ENABLED"] = False
        overrides["HIGH_LEVEL_MODE_SELECTION_ENABLED"] = True
    if args.llm_target_only:
        overrides["HIGH_LEVEL_MODE_SELECTION_ENABLED"] = False
        overrides["HIGH_LEVEL_TARGET_SELECTION_ENABLED"] = True
    if args.concurrent_min_frames is not None:
        overrides["HIGH_LEVEL_CONCURRENT_MIN_FRAMES"] = args.concurrent_min_frames
    if args.one_request_per_concurrent_episode is not None:
        overrides["HIGH_LEVEL_ONE_REQUEST_PER_EPISODE"] = (
            args.one_request_per_concurrent_episode
        )
    with temporary_config(**overrides):
        return run_batch(args)


def run_batch(args):
    output_path = (
        Path(args.output).resolve() if args.output else config.PROJECT_ROOT / "logs" / "experiments"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "summary.csv"
    )
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".metadata.json").exists():
        raise SystemExit(f"Experiment already exists; choose a new --output: {output_path}")
    artifact_dir = output_path.parent / (output_path.stem + "_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    with temporary_config(
        DATA_LOG_DIR=str(artifact_dir / "data"),
        CONTROL_LOG_DIR=str(artifact_dir / "control"),
        VISUAL_LOG_DIR=str(artifact_dir / "visuals"),
        VIDEO_LOG_DIR=str(artifact_dir / "videos"),
        ANALYSIS_LOG_DIR=str(artifact_dir / "analysis"),
        LOG_RETENTION_ENABLED=False,
        LLM_CACHE_ENABLED=not args.no_llm_cache,
        CONTROLLER_KIND=args.controller,
    ):
        return _run_batch(args, output_path, artifact_dir)


def _run_batch(args, output_path, artifact_dir):
    rows = []
    seeds = select_seeds(args)
    selected_cases = select_cases(args)

    backend_metadata = {"source": args.controller}
    if args.controller in ("openai", "deepseek", "vllm"):
        try:
            probe = build_controller(args.controller)
        except RuntimeError as exc:
            raise SystemExit(f"Controller setup failed: {exc}") from None
        backend_metadata = backend_snapshot(probe)
        probe.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = write_experiment_metadata(output_path, args, seeds, selected_cases,
                                               backend_metadata, artifact_dir / "source")
    # Preserve every run in this batch even when it exceeds the usual retention limit.
    max_log_files = max(args.max_log_files, len(seeds) * len(selected_cases))
    print(
        f"Running {len(selected_cases)} cases x {len(seeds)} seeds "
        f"with max_steps={args.max_steps}, controller={args.controller}"
    )
    print(f"Seeds: {seeds}")
    print(f"Results: {output_path}")
    for seed in seeds:
        for case in selected_cases:
            print(
                f"seed={seed} {case['case']}: {case['purpose']} "
                f"(A={case['num_abnormal_a']}, B={case['num_abnormal_b']}, C={case.get('num_abnormal_c', 0)}, "
                f"repair={case['repair_enabled']})"
            )
            row = run_case(
                case,
                seed=seed,
                max_steps=args.max_steps,
                max_log_files=max_log_files,
                save_visuals=args.visuals,
                save_video=args.video,
                save_analysis=args.analysis,
                controller_kind=args.controller,
                llm_cache_enabled=not args.no_llm_cache,
            )
            rows.append(row)
            write_summary(rows, output_path)
            print(
                f"  -> success={row['Success']}, frames={row['Frames']}, "
                f"final_dist={row['FinalDist']}, "
                f"danger_frames={row['DangerFrames']}, ts={row['Timestamp']}"
            )

    print(f"Summary saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
