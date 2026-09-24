import argparse
import time
from environment import SwarmEnv
from visualizer import Visualizer
from detector import Detector
from plot_analysis import plot_metrics_from_files
import config
from datetime import datetime
import os
from high_level_controller import build_controller

def parse_args():
    parser = argparse.ArgumentParser(description="Run one shepherding simulation.")
    parser.add_argument(
        "--controller",
        choices=("rule", "mock-llm", "openai", "deepseek"),
        default=config.CONTROLLER_KIND,
        help="Control architecture. API providers use asynchronous Responses calls.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode_msg = f"Fixed Mode (Seed {config.RANDOM_SEED})" if config.USE_FIXED_SEED else "Random Mode"
    print(f"Initializing simulation environment... ({mode_msg}, controller={args.controller})")
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env.save_data_to_parquet(timestamp)
    env.save_control_to_parquet(timestamp)
    
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
