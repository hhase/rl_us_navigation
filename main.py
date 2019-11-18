import io
import os
import sys
import PIL
import time
import torch
import random
import argparse
import numpy as np
from PIL import Image
from itertools import count
import torch.optim as optim
from datetime import datetime
import utils.DQN_utils as dqn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import utils.visualization as vis
import torchvision.transforms as T
import utils.memory_opt as mem_opt
from utils.ResNet18 import resnet18
import utils.environment_utils as env
import utils.utility_functions as fun
from tensorboardX import SummaryWriter
from utils.utility_functions import str2bool
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

if __name__ == '__main__':
    # ==================================================================================================#
    # PARSER SETUP AND PARSING                                                                          #
    # ==================================================================================================#
    parser = argparse.ArgumentParser(description='rl_usnavi')
    parser.add_argument('--run_name',           type=str,       default='prio_replay', metavar='str',  help='Run name')
    parser.add_argument('--steps_x',            type=int,       default=11,     metavar='X',    help='Grid steps on the x-axis (default: 5)')
    parser.add_argument('--steps_y',            type=int,       default=15,     metavar='Y',    help='Grid steps on the y-axis (default: 20)')
    parser.add_argument('--goal_x_coord',       type=int,       default=5,      metavar='x',    help='X-component of goal state (default: 2)')
    parser.add_argument('--goal_y_coord',       type=int,       default=0,      metavar='y',    help='Y-component of goal state (default: 10)')
    parser.add_argument('--margin_x',           type=int,       default=0,      metavar='x',    help='Margins on X to eliminate boarder frames (default: 5)')
    parser.add_argument('--margin_y',           type=int,       default=0,      metavar='y',    help='Margins on Y to eliminate boarder frames (default: 20)')
    parser.add_argument('--priority_exponent',  type=float,     default=0.5,    metavar='w',    help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority_weight',    type=float,     default=0.1,    metavar='b',    help='Importance sampling exponent (beta)')
    parser.add_argument('--reward_correct_nop', type=float,     default=1.0,    metavar='R',    help='Agent reward for stopping correctly (default: 1.0)')
    parser.add_argument('--reward_false_nop',   type=float,     default=-0.25,  metavar='R',    help='Agent reward for stopping wrongly (default: -0.25)')
    parser.add_argument('--reward_border',      type=float,     default=-0.1,   metavar='R',    help='Agent reward for stepping outside the grid (default: -0.2)')
    parser.add_argument('--reward_move_closer', type=float,     default=0.05,   metavar='R',    help='Agent reward for moving closer (default: 0.05)')
    parser.add_argument('--reward_move_away',   type=float,     default=-0.1,   metavar='R',    help='Agent reward for moving away (default: -0.1)')
    parser.add_argument('--batch_size_train',   type=int,       default=4,      metavar='N',    help='Input batch-size for training (default: 64)')
    parser.add_argument('--episodes',           type=int,       default=1000,   metavar='N',    help='Amount of episodes used for training (default: 1000)')
    parser.add_argument('--lr',                 type=float,     default=0.001,  metavar='LR',   help='Learning rate (default: 0.001)')
    parser.add_argument('--max_time_steps',     type=int,       default=50,     metavar='N',    help='Amount of steps the agent has to find a goal state in each episode(default: 50)')
    parser.add_argument('--log_interval',       type=int,       default=50,     metavar='N',    help='Episodes between logging (default: 50)')
    parser.add_argument('--memory_size',        type=int,       default=128,    metavar='M',    help='Size of replay memory buffer (default: 1.000.000)')
    parser.add_argument('--min_memory_size',    type=int,       default=128,    metavar='M',    help='Minimum amount of transitions in the replay memory for training (default: 1.000)')
    parser.add_argument('--target_update',      type=int,       default=20,     metavar='TU',   help='Target-update frequency in episodes (default: 20)')
    parser.add_argument('--target_update_lr',   type=float,     default=0.001,  metavar='LR',   help='Target-update frequency in episodes (default: 20)')
    parser.add_argument('--eps_start',          type=float,     default=1.0,    metavar='e',    help='Initial epsilon value for e-greedy policy (default: 1.0)')
    parser.add_argument('--eps_end',            type=float,     default=0.01,   metavar='e',    help='Final epsilon value for e-greddy policy (default: 0.01)')
    parser.add_argument('--eps_decay',          type=float,     default=0.0001, metavar='e',    help='Decay rate for epsilon value for e-greedy policy (default: 0.0001)')
    parser.add_argument('--eps_decay_begin',    type=int,       default=2500,   metavar='e',    help='Timestep where epsilon begins to decay (default: 15000)')
    parser.add_argument('--gamma',              type=float,     default=0.999,  metavar='g',    help='Discount factor (default: 0.999)')
    parser.add_argument('--prioritized_replay', type=str2bool,  default=True,                   help='Activates Prioritized Replay training (default: False)')
    parser.add_argument('--soft_target_update', type=str2bool,  default=False,                   help='Activates Prioritized Replay training (default: False)')
    parser.add_argument('--dueling_dqn',        type=str2bool,  default=True,                   help='Activates Prioritized Replay training (default: False)')
    parser.add_argument('--noisy_network',      type=str2bool,  default=False,                   help='Activates Noisy Network Layers (default: False)')
    parser.add_argument('--no-cuda',            action='store_true',                            help='disables CUDA training')
    parser.add_argument('--seed',               type=int,   default=1,      metavar='S',        help='random seed (default: 1)')
    args = parser.parse_args()
    print("Args parsed!")

    # ==================================================================================================#
    # BOOL VALUES TO ENABLE LOGGING AND MODEL LOADING/SAVING                                            #
    # ==================================================================================================#
    start = time.time()
    args.cuda =             not args.no_cuda and torch.cuda.is_available()
    cluster =               True if os.getenv('POLYAXON_RUN_OUTPUTS_PATH') else False
    prioritized_replay =    args.prioritized_replay
    soft_target_update =    args.soft_target_update
    dueling_dqn =           args.dueling_dqn
    noisy_network =         args.noisy_network
    log_progress =          True
    save_frames =           False
    save_model =            False
    load_model =            True

    print('prioritized_replay: ' + str(prioritized_replay))
    print('soft_target_update: ' + str(soft_target_update))
    print('dueling_dqn: '        + str(dueling_dqn))
    print('noisy_network: '      + str(noisy_network))

    # ==================================================================================================#
    # PATHS SETUP                                                                                       #
    # ==================================================================================================#
    run_name = args.run_name

    if cluster:
        data_paths = get_data_paths()
        patient_path = data_paths['data1'] + "/HHase_Robotic_RL/NAS_Sacrum_Scans/Patient_files/"
        patient_data_path = data_paths['data1'] + "/HHase_Robotic_RL/NAS_Sacrum_Scans/"
        load_model_path = data_paths['data1'] + "/HHase_Robotic_RL/Models/model_best.pth"
        output_path = get_outputs_path()
        tensorboard_path = get_outputs_path()
        experiment = Experiment()
    else:
        patient_path = "./../Data/Patient_files/"
        patient_data_path = "./../Data/"
        output_path = './'
        tensorboard_path = './runs/'
        load_model_path = "./../Data/pretrained_model/model_best.pth"
        model_save_path = output_path + "/models/{}.pt".format(run_name)

    #load_model_path = output_path + "/models/{}.pt".format(run_name)
    datetime = datetime.now()
    tensorboard_name = 'Nov' + datetime.strftime("%d_%H-%M-%S") + '_Rachet-' + run_name

    # ==================================================================================================#
    # PARAMETER INITIALIZATION                                                                          #
    # ==================================================================================================#

    torch.cuda.manual_seed(np.random.randint(1, 10000)) if args.cuda else torch.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.benchmark = True
    sys.setrecursionlimit(2000)

    # PARAMETER SETTING
    batch_size =        args.batch_size_train
    lr =                args.lr
    max_time_steps =    args.max_time_steps
    memory_size =       args.memory_size
    min_memory_size =   args.min_memory_size
    num_episodes =      args.episodes
    target_update =     args.target_update
    target_update_lr =  args.target_update_lr
    gamma =             args.gamma

    beta_update_step = (1.0 - args.priority_weight) / num_episodes

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    # ==================================================================================================#
    # LOGGING INITIALIZATION                                                                            #
    # ==================================================================================================#
    if log_progress:
        writer = SummaryWriter(tensorboard_path + '/' + tensorboard_name)

        writer.add_text('Goal_coords', 'Goal coordinates: [{}, {}]'.format(args.goal_x_coord,args.goal_y_coord))
        writer.add_text('Rewards', 'Rewards for: Moving closer = {} | '.format(args.reward_move_closer) + \
                        'Moving away = {} | '.format(args.reward_move_away) + \
                        'Colliding with a border = {} | '.format(args.reward_border) + \
                        'Correct NOP = {} | '.format(args.reward_correct_nop) + \
                        'Incorrect NOP = {}'.format(args.reward_false_nop))
        writer.add_text('Batch_size', 'Batch_size = {}'.format(batch_size))
        writer.add_text('Replay Memory size', 'Replay memory size = {}'.format(memory_size))
        writer.add_text('Learning rate', 'Learning rate = {}'.format(lr))
        writer.add_text('DQN Settings', "Dueling DQN: {} | Noisy Net: {} | Prioritized Replay: {} | Soft Target Update: {}"\
                        .format(dueling_dqn, noisy_network, prioritized_replay, soft_target_update))

        # ==================================================================================================#
        # DATA LOADING                                                                                      #
        # ==================================================================================================#
        train_patients = []
        test_patients = []
        patient_counter = 0
        goals = []

        for file in os.listdir(patient_path):
            patient = env.load_patient(patient_path + file, patient_data_path, args, cluster)

            if patient.id % 5 == 0:
                test_patients.append(patient)
            train_patients.append(patient)

        current_patient = random.choice(train_patients)

        print("Training patients: " + ", ".join([patient.name for patient in train_patients]))
        print("Testing patients: " + ", ".join([patient.name for patient in test_patients]))
        print("Patients set up correctly!")

    # ==================================================================================================#
    # ENVIRONMENT SETUP                                                                                 #
    # ==================================================================================================#

    if noisy_network:
        strategy = dqn.GreedyStrategy(args)
    else:
        strategy = dqn.EpsilonGreedyStrategy(args)

    agent = dqn.Agent(strategy, current_patient.grid.num_actions_available(), device)

    memory = mem_opt.ReplayMemory(args)
    print("Prioritized Replay memory set up correctly!") if prioritized_replay else print("Replay memory set up correctly!")

    if dueling_dqn:
        policy_net = resnet18().to(device)
        target_net = resnet18().to(device)
        target_net.eval()
        #target_net.load_state_dict(policy_net.state_dict())
        #policy_net = dqn.Dueling_DQN(args, current_patient.grid.get_frame_dims(), current_patient.grid.num_actions_available()).to(device)
        #target_net = dqn.Dueling_DQN(args, current_patient.grid.get_frame_dims(), current_patient.grid.num_actions_available()).to(device)
        print("Dueling-DQN set up correctly!")
    else:
        policy_net = dqn.DQN(args, current_patient.grid.get_frame_dims(), current_patient.grid.num_actions_available()).to(device)
        target_net = dqn.DQN(args, current_patient.grid.get_frame_dims(), current_patient.grid.num_actions_available()).to(device)
        print("DQN set up correctly!")

    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    #optimizer = optim.SGD(params=policy_net.parameters(), lr=lr, momentum=0.9)

    if load_model:
        checkpoint = torch.load(load_model_path)
        policy_net.load_state_dict(checkpoint['state_dict'])

        fig, correctness = vis.display_policy(patient=current_patient, net=policy_net, target=target_net, dueling=args.dueling_dqn, device=device)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = T.ToTensor()(image)
        writer.add_image('Initialization/Policy grid at initial episode', image, 0)
        plt.close(fig)

        optimizer.load_state_dict(checkpoint['optimizer'])

        policy_net.train() # for resuming training
        #policy_net.eval() # for inference

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    print("Target Network set up correctly!")
    end = time.time()
    print("Setup time = {}".format(end-start))

    # ==========================================================================================#
    # FILL REPLAY MEMORY WITH RANDOM TRANSITIONS TO START TRAINING                              #
    # ==========================================================================================#

    print("Filling replay memory with random samples!")
    start = time.time()
    for i in range(min_memory_size):
        if i % 50 == 0:
            current_patient.grid.border_collisions = 0
            current_patient = random.choice(train_patients)
            print("{} - changing to {}".format(i, current_patient.name))
        current_patient.grid.reset_grid()
        state = current_patient.grid.get_state()
        state_idx = torch.from_numpy(current_patient.grid.current_bin.coords)
        action = agent.select_action(state, policy_net, 0)
        reward = current_patient.grid.take_action(action)
        next_state = current_patient.grid.get_state()
        memory.append(state, action, reward, next_state)
    end = time.time()
    current_patient.grid.border_collisions = 0
    print("Memory loadad in {}s".format(end - start))

    # ==================================================================================================#
    # TRAINING LOOP - 1 EPISODE = 1 RUN OF n TIMESTEPS                                                  #
    # ==================================================================================================#
    # print("Starting to train!")
    for episode in range(num_episodes):

        # ==============================================================================================#
        # PARAMETER RESET                                                                               #
        # ==============================================================================================#
        current_patient.grid.reset_grid()
        state = current_patient.grid.get_state()
        current_patient.grid.border_collisions = 0
        correctness = 0
        accum_reward = 0
        average_loss = 0
        nop_correct_count = 0
        current_patient.grid.current_reward = 0
        nop_incorrect_count = 0
        positive_rewards_count = 0
        action_list = []
        reward_list = []

        for timestep in count(start=1, step=1):
            # ==========================================================================================#
            # AGENT - ENVIRONMENT INTERACION WITH SAVING TRANSITION IN THE REPLAY MEMORY                #
            # ==========================================================================================#

            x, y = current_patient.grid.current_bin.coords
            state_idx = torch.from_numpy(current_patient.grid.current_bin.coords)
            current_patient.grid.visits[x, y] += 1
            action = agent.select_action(state, policy_net, episode * max_time_steps + timestep)
            reward = current_patient.grid.take_action(action)
            next_state = current_patient.grid.get_state()
            memory.append(state, action, reward, next_state)

            action_list.append(action.detach().cpu().numpy())
            reward_list.append(reward.detach().cpu().numpy())

            state = next_state
            if reward > current_patient.grid.reward_dict['further']:
                positive_rewards_count += 1
            if action == 4:
                if reward == 1.0:
                    nop_correct_count += 1
                else:
                    nop_incorrect_count += 1
            accum_reward += reward

            # ==========================================================================================#
            # DQN OPTIMIZATION STEP                                                                     #
            # ==========================================================================================#
            start = time.time()

            if policy_net.noisy:
                policy_net.reset_noise()

            tree_idxs, states, actions, rewards, next_states, weights = memory.sample(batch_size)
            current_q_values = dqn.QValues.get_current_q_values(policy_net, states, actions)
            terminals = fun.check_nop(actions, rewards) + fun.check_nop(actions, rewards, true_positive=False)

            if np.sum(terminals) == batch_size:
                target_q_values = rewards
            else:
                next_actions = dqn.QValues.get_next_actions(policy_net, next_states)
                next_q_values = dqn.QValues.get_next_q_values(target_net, next_states, next_actions, terminals)
                target_q_values = (next_q_values * gamma) + rewards

            if prioritized_replay:
                TD_errors = dqn.get_TD_errors(policy_net, target_net, states, actions, rewards, next_states, gamma)
                memory.update_priorities(tree_idxs, TD_errors)

            if prioritized_replay:
                loss = torch.sum((target_q_values - current_q_values) ** 2 / batch_size, dim=1)
                optimizer.zero_grad()
                (weights * loss).mean().backward()
                average_loss += (weights * loss).mean().detach().cpu().numpy()/max_time_steps
                optimizer.step()
            else:
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                average_loss += loss.detach().cpu().numpy()/max_time_steps
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            end = time.time()
            print("Done with step nr: " + str(timestep) + " in {}s ".format(end-start))

            # ==========================================================================================#
            # POLICY GRAPH FOR AN INDIVIDUAL TIME STEP FOR DEBUGGING OR ANIMATION (stored in "./imgs/") #
            # ==========================================================================================#
            if not cluster and save_frames:
                fig, correctness = vis.display_policy(patient=patient, net=policy_net, target=target_net,
                                                      dueling=args.dueling_dqn, device=device)
                agent_position = state_idx.detach().cpu().numpy()
                action_taken = current_patient.grid.action_space[action.detach().cpu().numpy()[0]]
                reward_obtained = reward.detach().cpu().numpy()
                fig.suptitle("Episode {} - Timestep {}\n".format(episode, timestep) + \
                             "Agent position: {} - ".format(agent_position) + \
                             "Action taken: {} - ".format(action_taken) + \
                             "Reward obtained: {}".format(reward_obtained))
                im_name = output_path + "imgs/{}-episode_{}-timestep_{}.png".format(run_name, episode, timestep)
                plt.savefig(im_name, facecolor='w', edgecolor='w', orientation='portrait', pad_inches=0.1)
                plt.close(fig)

            # ==============================================================================================#
            # LOGGING RELEVANT VALUES AT THE END OF THE EPISODE                                             #
            # ==============================================================================================#
            if timestep == args.log_interval:
                start = time.time()
                if log_progress:
                    writer.add_scalar('Training/Cumulated Reward', accum_reward, episode)
                    writer.add_scalar('Training/NOP True Positives', nop_correct_count, episode)
                    writer.add_scalar('Training/NOP False Positives', nop_incorrect_count, episode)
                    writer.add_scalar('Training/Positive reward ratio', positive_rewards_count/max_time_steps, episode)
                    writer.add_scalar('Training/Epsilon value', strategy.get_exploration_rate(episode * max_time_steps + timestep), episode)
                    train_correctness = 0
                    for patient in train_patients:
                        fig, correctness, state_vals, advantages = vis.display_policy(patient=patient, net=policy_net, dueling=args.dueling_dqn, device=device, action_state_values=True)
                        plt.close(fig)
                        train_correctness += correctness/len(train_patients)
                    print("Training correctness for {} is {}".format(", ".join([patient.name for patient in train_patients]), train_correctness))
                    writer.add_scalar('Training/Policy correctness - train', train_correctness, episode)
                    writer.add_scalar('Training/State values - mean', state_vals[0], episode)
                    writer.add_scalar('Training/State values - std', state_vals[1], episode)
                    writer.add_scalar('Training/Advantage values - mean', advantages[0], episode)
                    writer.add_scalar('Training/Advantage values - std', advantages[1], episode)
                    test_correctness = 0
                    for patient in test_patients:
                        fig, correctness = vis.display_policy(patient=patient, net=policy_net, dueling=args.dueling_dqn, device=device)
                        plt.close(fig)
                        test_correctness += correctness/len(test_patients)
                    print("Testing correctness for {} is {}".format(", ".join([patient.name for patient in test_patients]), test_correctness))
                    writer.add_scalar('Testing/Policy correctness - test', test_correctness, episode)
                    writer.add_scalar('Training/Loss', average_loss, episode)
                    writer.add_scalar('Training/Border collisions', current_patient.grid.border_collisions, episode)

                    if cluster:
                        experiment.log_metrics(step=episode,
                                               NOP_true_positives=nop_correct_count,
                                               positive_reward_ratio=positive_rewards_count/max_time_steps,
                                               epsilon_value = strategy.get_exploration_rate(episode * max_time_steps + timestep),
                                               loss=average_loss,
                                               train_correctness=train_correctness,
                                               test_correctness=test_correctness)

                if save_model:
                    torch.save({
                        'episode': episode,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, model_save_path)

                current_patient = random.choice(train_patients)
                print("Training now on {}".format(current_patient.name))
                end = time.time()
                print("Logging time = {}".format(end - start))
                break

        print(positive_rewards_count/max_time_steps, nop_incorrect_count)
        [print(elem) for elem in zip(action_list, reward_list)]

        if prioritized_replay:
            memory.update_beta(beta_update_step)

        print("Episode number {} done!".format(episode))

        if log_progress and episode % 20 == 0:
            fig, correctness = vis.display_policy(patient=patient, net=policy_net, target=target_net, dueling=args.dueling_dqn, device=device)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = T.ToTensor()(image)
            writer.add_image('Policy grid at episode {}'.format(episode), image, episode)
            plt.close(fig)

        if soft_target_update:
            for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(target_update_lr * local_param.data + (1.0 - target_update_lr) * target_param.data)

        if not soft_target_update and episode % target_update == 0 and episode > 0:
            print("Updating target net!")
            target_net.load_state_dict(policy_net.state_dict())
