import argparse
import json
import time
from environment import SwarmEnv
from visualizer import Visualizer
from detector import Detector
from plot_analysis import plot_metrics_from_files
import config
from datetime import datetime
import os
from high_level_controller import build_controller

from config import SCENARIOS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one shepherding simulation.")
    parser.add_argument(
        "--controller",
        choices=("rule", "mock-llm", "openai", "deepseek"),
        default=config.CONTROLLER_KIND,
        help="Control architecture. API providers use asynchronous Responses calls.",
    )
    parser.add_argument(
        "--scenario", choices=("config", *SCENARIOS), default="config",
        help="Sheep population only; independent of the controller and repair setting.",
    )
    parser.add_argument("--abnormal-a", type=int, help="Override the number of Type A sheep.")
    parser.add_argument("--abnormal-b", type=int, help="Override the number of Type B sheep.")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="Fixed random seed for matched comparisons.")
    seed_group.add_argument("--random-seed", action="store_true", help="Random initial conditions.")
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument(
        "--repair", action=argparse.BooleanOptionalAction, default=config.REPAIR_ENABLED,
        help="Enable anomaly repair; --no-repair disables repair in either architecture.",
    )
    parser.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=config.SAVE_ANIMATION_MP4,
        help="Export MP4 after the simulation.",
    )
    args = parser.parse_args(argv)
    count_a, count_b = SCENARIOS.get(
        args.scenario, (config.NUM_ABNORMAL_A, config.NUM_ABNORMAL_B)
    )
    args.abnormal_a = count_a if args.abnormal_a is None else args.abnormal_a
    args.abnormal_b = count_b if args.abnormal_b is None else args.abnormal_b
    if args.abnormal_a < 0 or args.abnormal_b < 0:
        parser.error("Abnormal sheep counts must be nonnegative.")
    if args.abnormal_a + args.abnormal_b > config.NUM_SHEEP:
        parser.error(f"Type A + Type B cannot exceed NUM_SHEEP ({config.NUM_SHEEP}).")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295.")
    return args


def apply_run_settings(args):
    config.NUM_ABNORMAL_A = args.abnormal_a
    config.NUM_ABNORMAL_B = args.abnormal_b
    config.MAX_STEPS = args.max_steps
    config.REPAIR_ENABLED = args.repair
    config.SAVE_ANIMATION_MP4 = args.video
    config.CONTROLLER_KIND = args.controller
    if args.seed is not None:
        config.USE_FIXED_SEED = True
        config.RANDOM_SEED = args.seed
    elif args.random_seed:
        config.USE_FIXED_SEED = False


def main(argv=None):
    args = parse_args(argv)
    apply_run_settings(args)
    mode_msg = f"Fixed Mode (Seed {config.RANDOM_SEED})" if config.USE_FIXED_SEED else "Random Mode"
    print(f"Initializing simulation environment... ({mode_msg}, controller={args.controller})")
    print(
        f"Sheep: {config.NUM_SHEEP}, A: {config.NUM_ABNORMAL_A}, "
        f"B: {config.NUM_ABNORMAL_B}, repair: {config.REPAIR_ENABLED}, "
        f"max frames: {config.MAX_STEPS}, video: {config.SAVE_ANIMATION_MP4}"
    )
    try:
        controller = build_controller(args.controller)
    except RuntimeError as exc:
        raise SystemExit(f"Controller setup failed: {exc}") from None
    env = SwarmEnv(controller=controller)
    detector = Detector(dispersion_threshold=config.DANGER_DISPERSION_THRESHOLD,
                          mean_spread_threshold=config.DANGER_MEAN_SPREAD_THRESHOLD,
                          stagnation_frames=config.DANGER_STAGNATION_FRAMES,
                          stagnation_threshold=config.DANGER_STAGNATION_THRESHOLD)
    vis = Visualizer(env)  
    vis.ax.set_title(
        f"{args.controller} | A={config.NUM_ABNORMAL_A}, B={config.NUM_ABNORMAL_B} "
        f"| seed={config.RANDOM_SEED if config.USE_FIXED_SEED else 'random'}"
    )

    step_count = 0

    print("The shepherd starts working!")
    while step_count < config.MAX_STEPS:
        # 1. run the detector to analyze the flock's state and record metrics
        alarms, metrics, report = detector.detect(env.sheep_pos, env.goal_pos)
        env.record_frame_metrics(metrics)
        if report['is_danger']:
            # to avoid spamming too many warnings, we only print every 10 frames when in danger
            if step_count % 10 == 0:
                print(f"🚨 [Frame {step_count}] Danger Warning! Type: {report['danger_type']}. Desc: {report['description']}")

        # 2. physics environment simulation one frame
        reached_goal = env.step(alarms)
        
        # 3. refresh the screen
        vis.render(env)
        if (
            args.controller in ("openai", "deepseek")
            and getattr(env.controller, "pending_request", None) is not None
        ):
            time.sleep(config.LLM_PENDING_POLL_SECONDS)

        # 4. check if the mission is successful
        if reached_goal:
            print(f"✅ Mission Accomplished! total cost: {step_count} frames.")
            break
            
        step_count += 1

    if step_count >= config.MAX_STEPS:
        print("⚠️ Maximum steps reached, simulation terminated.")

    # Export the recorded simulation data
    print("Exporting simulation data...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    env.save_data_to_parquet(timestamp)
    env.save_control_to_parquet(timestamp)
    settings = {
        key: value for key, value in vars(config).items()
        if key.isupper() and isinstance(value, (str, int, float, bool, type(None)))
    }
    with open(os.path.join(config.DATA_LOG_DIR, f"run_{timestamp}.json"), "w", encoding="utf-8") as metadata_file:
        json.dump({"scenario": args.scenario, "config": settings}, metadata_file, indent=2)
    
    if config.SAVE_TRAJECTORY_PNG:
        vis.save_trajectory(env, timestamp)
    if config.SAVE_ANIMATION_MP4:
        print("Saving animation mp4, this may take a while...")
        vis.save_animation_mp4(env, timestamp)
        
    print("Generating analysis charts...")
    parquet_path = os.path.join(config.DATA_LOG_DIR, f"data_{timestamp}.parquet")
    plot_metrics_from_files([parquet_path], config.ANALYSIS_LOG_DIR)

    print("✅ Data successfully saved.")

    env.close()
    vis.close()

if __name__ == "__main__":
    main()
