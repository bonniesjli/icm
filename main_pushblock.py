from utils import *
from ppo import *
from loggin import *
from torch.multiprocessing import Pipe
from mlagents.envs import UnityEnvironment
import numpy as np
import time
import sys

print("Python version:")
print(sys.version)
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")
arg1 = sys.argv[1]
if arg1 == "linux":
    env = UnityEnvironment(file_name = "../../bonnie_envs/pushblock_linux/pushblock.x86_64")
if arg1 == "window":
    env = UnityEnvironment(file_name = "../../bonnie_envs/pushblock_window/Unity Environment.exe")
    
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

def main(run, icm = True):
    """
    :param: (str) run
    :param: (bool) icm
    """
    env.reset()
    
    max_t = 6e4 #1e5
    t_horizon = 10
    t = 0
    num_worker = 32
    input_size = 210
    output_size = 5
    num_step = 256
    gamma = 0.99
    pre_obs_norm_step = 10000
    
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(1, input_size)
    discounted_reward = RewardForwardFilter(gamma)
    agent = PPOAgent(input_size, output_size, num_worker,num_step, gamma)

    steps = 0
    next_obs = []
    print('Start to initialize observation normalization ...')
    while steps < pre_obs_norm_step:
        steps += num_worker
        actions = np.random.randint(output_size, size=num_worker)
        env_info = env.step(actions)[default_brain]
        obs = env_info.vector_observations
        for o in obs:
            next_obs.append(o)
        print('initializing...:', steps, '/', pre_obs_norm_step)
    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs)
    print('End to initialize')

    global_update = 0
    global_step = 0
    sample_i_rall = 0
    sample_episode = 0
    sample_env_idx = 0
    sample_rall = 0
    states = np.zeros([num_worker, input_size])
    int_coef = 0.5
    large_scale_version = True

    LOG = Logging(run)
    LOG.create("score")
    LOG.create("full_score")
    LOG.create("intrinsic_reward")

    agent_r = np.zeros(num_worker)
    buffer_r = np.zeros(num_worker)

    while t <= max_t:
        total_state, total_reward, total_done, total_next_state, \
        total_action, total_int_reward, total_next_obs, total_values,\
        total_policy, total_combine_reward = [], [], [], [], [], [], [], [], [], []
        global_step += (num_step * num_worker)
        global_update += 1

        for _ in range(num_step):
            t += 1
            actions, value, policy = agent.get_action((np.float32(states) - obs_rms.mean)/np.sqrt(obs_rms.var))

            env_info = env.step(actions)[default_brain]

            next_states, rewards, dones, real_dones, next_obs = [], [], [], [], []

            obs = env_info.vector_observations
            reward = env_info.rewards
            # reward = np.clip(reward, 0, 1)
            done = env_info.local_done

            # MY LOGGER
            agent_r += reward
            for j, d in enumerate(done):
                    if done[j]:
                        buffer_r[j] = agent_r[j]
                        agent_r[j] = 0

            for o, r, d in zip(obs, reward, done):
                next_states.append(o)
                rewards.append(r)
                dones.append(d)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            
            if icm:
                intrinsic_reward = agent.icm.compute_intrinsic_reward(
                                    (states - obs_rms.mean)/np.sqrt(obs_rms.var),
                                    (next_states - obs_rms.mean)/np.sqrt(obs_rms.var),
                                    actions)
                # print (intrinsic_reward)
                LOG.log("intrinsic_reward", intrinsic_reward)
                intrinsic_reward = np.hstack(intrinsic_reward)
                combine_reward = (1-int_coef) * rewards + int_coef * intrinsic_reward
                
            if not icm: 
                intrinsic_reward = np.zeros(num_worker)
                combine_reward = rewards

    #         sample_i_rall += intrinsic_reward[sample_env_idx]
    #         sample_rall += rewards[sample_env_idx]

            total_combine_reward.append(combine_reward)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_values.append(value)
            total_policy.append(policy)

            states = next_states[:, :]

    #         if dones[sample_env_idx]:
    #             sample_episode += 1
    #             sample_i_rall = 0
    #             sample_rall = 0

    #             print("[Episode {}] rall: {}  i_: {}".format(
    #                     sample_episode, sample_rall, sample_i_rall))

    #             sample_i_rall = 0
    #             sample_rall = 0

            if t % 1000 == 0:
                LOG.log("score", np.mean(buffer_r))
                LOG.log("full_score", buffer_r)

            if t < 10000 and t % 2000 == 0:
                print('\rTimeStep {}\tAverage Score: {:.2f}'.format(t, LOG.mean("score", t_horizon)))
                LOG.save_data()

            if t >= 10000 and t % 10000 == 0:
                print('\rTimeStep {}\tAverage Score: {:.2f}'.format(t, LOG.mean("score", t_horizon)))
                torch.save(agent.model, 'model.pt')
                LOG.save_data()



        _, value, _ = agent.get_action((np.float32(states) - obs_rms.mean) / np.sqrt(obs_rms.var))
        total_values.append(value)

        total_state = np.stack(total_state).transpose([1, 0, 2]).reshape([-1, input_size])
        total_next_state = np.stack(total_next_state).transpose([1, 0, 2]).reshape([-1, input_size])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_reward = np.stack(total_reward).transpose()
        total_done = np.stack(total_done).transpose()
        total_values = np.stack(total_values).transpose()
        total_logging_policy = np.vstack(total_policy)
        total_combine_reward = np.stack(total_combine_reward).transpose()

        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                             total_combine_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        total_int_reward /= np.sqrt(reward_rms.var)
        total_combine_reward /= np.sqrt(reward_rms.var)

        if large_scale_version: flag = np.zeros_like(total_combine_reward)
        else: flag = total_done

        target ,adv = make_train_data_icm(total_combine_reward, flag, total_values, gamma, num_step, num_worker)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        print('training')
        agent.train_model((np.float32(total_state) - obs_rms.mean )/ np.sqrt(obs_rms.var),
                            (np.float32(total_next_state) - obs_rms.mean) / np.sqrt(obs_rms.var),
                            target, total_action,
                            adv, total_policy)
    LOG.save_data()
    LOG.visualize("score")
    
main("run1")
main("run2", False)
main("run3")
main("run4")
main("run5", False)
main("run6", False)
