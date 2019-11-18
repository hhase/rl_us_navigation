import numpy as np
import time
import matplotlib.pyplot as plt
import utils.DQN_utils as dqn


def get_state_values(grid=None, net=None, dueling=False, device=None):
    if dueling:
        net.eval()
        net.get_state_value = True
        save_position = grid.current_bin.coords
        grid_size = grid.bin_array.shape
        state_vals = np.zeros(grid_size)

        for col in range(grid_size[1]):
            for row in range(grid_size[0]):
                grid.set_grid(np.array([row, col]))
                state = grid.get_state(frame_pos=0)
                state_value = net(state).to(device)
                state_vals[row, col] = state_value.detach().cpu().numpy()

        state_vals = state_vals - np.max(state_vals)
        state_vals = (np.sign(state_vals) * state_vals/np.min(state_vals) + 1.001) * 100

        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        rows = [i % grid_size[1] for i in range(int(grid_size[0] * grid_size[1]))]
        cols = [int(i / grid_size[1]) for i in range(int(grid_size[0] * grid_size[1]))]

        ax.set_title("State-values")
        ax.scatter(rows, cols, s=state_vals)

        grid.set_grid(save_position)
        net.get_state_value = False
        net.train()
        plt.close(fig)
    else:
        state_vals = np.zeros_like(grid.bin_array.shape)
        state_vals = (state_vals + 1) * 100
    return state_vals.reshape(-1)


def display_policy(patient=None, net=None, target=None, dueling=False, action_state_values=False, device=None):
    net.eval()
    name = patient.name
    grid = patient.grid
    state_vals = np.zeros(grid.bin_array.shape)
    save_position = grid.current_bin.coords
    grid.set_grid(save_position)
    correctness = 0
    grid.plotting = True

    u_dict = {0: 0, 1:  0, 2: -1, 3: 1, 4: 0}
    v_dict = {0: 1, 1: -1, 2:  0, 3: 0, 4: 0}
    grid_size = grid.bin_array.shape
    goal = grid.goal_coords[-1]
    position = grid.current_bin.coords
    plot_strat = dqn.GreedyStrategy([])
    plot_agent = dqn.Agent(plot_strat, grid.num_actions_available(), device)

    if action_state_values:
        state_values = np.zeros([grid_size[0]*grid_size[1]])
        advantage_values = np.zeros([grid_size[0]*grid_size[1], 5])

    if target:
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        fig.set_figheight(7)
        fig.set_figwidth(16)
    else:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.set_figheight(7)
        fig.set_figwidth(11)

    rows = [i % grid_size[1] for i in range(int(grid_size[0]*grid_size[1]))]
    cols = [int(i / grid_size[1]) for i in range(int(grid_size[0]*grid_size[1]))]

    ax[0].set_yticks(np.arange(0, 11, 1.0))
    ax[0].set_xticks(np.arange(0, 15, 1.0))

    U = np.zeros(grid_size)
    V = np.zeros(grid_size)
    start = time.time()
    for col in range(U.shape[1]):
        for row in range(U.shape[0]):
            grid.set_grid(np.array([row, col]))
            state = grid.get_state(frame_pos=0)
            action = plot_agent.select_action(state, net, 0)
            action_val = action.detach().cpu().numpy()[0]
            reward = grid.take_action(action)
            if action_state_values:
                value, advantage = net.get_state_action_values(state)
                state_values[col*U.shape[0] + row] = value.detach().cpu().numpy()
                advantage_values[col*U.shape[0] + row, :] = advantage.detach().cpu().numpy()

            if dueling:
                state_value = net(state, state_value=True).to(device)
                state_vals[row, col] = state_value.detach().cpu().numpy()

            if reward >= grid.reward_dict['closer']:
                correctness += 1

            U[row, col] = u_dict[action_val]
            V[row, col] = v_dict[action_val]

    end = time.time()
    #print("Iterate through whole grid in {}s".format(end-start))
    if dueling:
        state_vals = state_vals - np.max(state_vals)
        state_vals = (np.sign(state_vals) * state_vals / np.min(state_vals) + 1.001) * 100
    else:
        state_vals = (state_vals + 1) * 100
        state_vals.reshape(-1)

    ax[0].scatter(rows, cols, s=state_vals)
    ax[0].scatter(position[1], position[0], marker='*', c="red", s=800)
    for goal in grid.goal_coords:
        ax[0].scatter(goal[1], goal[0], marker='s', c="green", s=800)

    ax[0].quiver(rows, cols, U, V, scale=30)

    correctness = correctness / (U.shape[0] * U.shape[1])
    ax[0].set_title("DQN policy | Patient: {} | correctness: {:.2f}".format(name, correctness))

    heat_map = grid.visits/(np.max(grid.visits) + 1) * 255
    heat_map = heat_map.astype(int)
    ax[1].set_title("Visited states")
    ax[1].matshow(heat_map, cmap='Reds')

    if target:
        # Plot target network
        ax[2].set_title("Target network policy")
        ax[2].scatter(rows, cols, s=50)
        ax[2].set_yticks(np.arange(0, 11, 1.0))
        ax[2].set_xticks(np.arange(0, 15, 1.0))
        ax[2].xaxis.tick_top()

        # Plot goal
        ax[2].scatter(goal[1], goal[0], marker='s', c="green", s=1000)

        U = np.zeros(grid_size)
        V = np.zeros(grid_size)
        for col in range(U.shape[1]):
            for row in range(U.shape[0]):
                grid.set_grid(np.array([row, col]))
                state = grid.get_state(frame_pos=0)
                action = plot_agent.select_action(state, target, 0)
                action_val = action.detach().cpu().numpy()[0]

                U[row, col] = u_dict[action_val]
                V[row, col] = v_dict[action_val]

        ax[2].quiver(rows, cols, U, V, scale=30)

    net.train()
    grid.set_grid(save_position)
    grid.plotting = False

    if action_state_values:
        state_values = [state_values.mean(), state_vals.std()]
        advantage_values = [advantage_values.mean(), advantage_values.std()]
        return fig, correctness, state_values, advantage_values
    else:
        return fig, correctness
