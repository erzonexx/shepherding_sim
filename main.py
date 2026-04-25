from environment import SwarmEnv
from visualizer import Visualizer
import config

def main():
    print("Initializing simulation environment...")
    env = SwarmEnv()
    vis = Visualizer()  

    step_count = 0

    print("The shepherd starts working!")
    while step_count < config.MAX_STEPS:
        # 1. physics environment simulation one frame
        reached_goal = env.step()
        
        # 2. refresh the screen
        vis.render(env)

        # 3. check if the mission is successful
        if reached_goal:
            print(f"✅ Mission Accomplished! total cost: {step_count} frames。")
            break
            
        step_count += 1

    if step_count >= config.MAX_STEPS:
        print("⚠️ Maximum steps reached, simulation terminated.")

    vis.close()

if __name__ == "__main__":
    main()