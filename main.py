from environment import SwarmEnv
from visualizer import Visualizer
import config
from datetime import datetime

def main():
    mode_msg = "Fixed Mode (Seed 42)" if config.USE_FIXED_SEED else "Random Mode"
    print(f"Initializing simulation environment... ({mode_msg})")
    env = SwarmEnv()
    vis = Visualizer(env)  

    step_count = 0

    print("The shepherd starts working!")
    while step_count < config.MAX_STEPS:
        # 1. physics environment simulation one frame
        reached_goal = env.step()
        
        # 2. refresh the screen
        vis.render(env)

        # 3. check if the mission is successful
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
    
    if config.SAVE_TRAJECTORY_PNG:
        vis.save_trajectory(env, timestamp)
    if config.SAVE_ANIMATION_MP4:
        print("Saving animation mp4, this may take a while...")
        vis.save_animation_mp4(env, timestamp)
        
    print("✅ Data successfully saved.")

    vis.close()

if __name__ == "__main__":
    main()