import matplotlib.pyplot as plt
import config

class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(0, config.SPACE_SIZE)
        self.ax.set_ylim(0, config.SPACE_SIZE)
        self.ax.set_title("Swarm Shepherding Simulation")

        # draw the goal area (golden star)
        self.ax.plot(config.GOAL_POS[0], config.GOAL_POS[1], 'y*', markersize=18, label='Goal')
        
        # initialize plot objects for dynamic update
        self.sheep_plot, = self.ax.plot([], [], 'bo', markersize=6, label='Sheep')
        self.dog_plot, = self.ax.plot([], [], 'rs', markersize=9, label='Dog')

        self.ax.legend(loc='upper right')
        plt.ion() # enable interactive mode, allow dynamic refresh

    def render(self, env):
        # update the data of the plot objects instead of redrawing
        self.sheep_plot.set_data(env.sheep_pos[:, 0], env.sheep_pos[:, 1])
        self.dog_plot.set_data([env.dog_pos[0]], [env.dog_pos[1]])

        plt.pause(0.01) # pause for a very short time to refresh the screen

    def close(self):
        plt.ioff()
        plt.show()