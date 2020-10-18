from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
    
    ######### this is the loss pattern generated##########   
    prob_transition_matrix_heavy = [
        [0.4,0.6,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.5,0.0,0.5,0.0,0.0,0.0,0.0,0.0],
        [0.45,0.0,0.0,0.55,0.0,0.0,0.0,0.0],
        [0.61,0.0,0.0,0.0,0.39,0.0,0.0,0.0],
        [0.37,0.0,0.0,0.0,0.0,0.63,0.0,0.0],
        [0.77,0.0,0.0,0.0,0.0,0.0,0.23,0.0],
        [0.89,0.0,0.0,0.0,0.0,0.0,0.0,0.11],
        [0.93,0.0,0.0,0.0,0.0,0.0,0.0,0.07]
    ]   
    def generate_loss_pattern_heavy(self, episode_length = 200, p_matrix = prob_transition_matrix_heavy):
        rand = th.rand((episode_length,3))
        loss_pattern = th.zeros((episode_length,3))
        for i in range(3):
            state = '0'   # '0' means no error, '1' means error
            for ind in range(episode_length):
                if (state == '0') and (rand[ind,i] <= p_matrix[0][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 1
                elif (state == '0') and (rand[ind,i] >= p_matrix[0][0]):
                    next_state = '1'
                    loss_pattern[ind,i] = 1
                elif (state == '1') and (rand[ind,i] <= p_matrix[1][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '1') and (rand[ind,i] >= p_matrix[1][0]): 
                    next_state = '2'
                    loss_pattern[ind,i] = 0
                elif (state == '2') and (rand[ind,i] <= p_matrix[2][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '2') and (rand[ind,i] >= p_matrix[2][0]): 
                    next_state = '3'
                    loss_pattern[ind,i] = 0
                elif (state == '3') and (rand[ind,i] <= p_matrix[3][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '3') and (rand[ind,i] >= p_matrix[3][0]): 
                    next_state = '4'
                    loss_pattern[ind,i] = 0
                elif (state == '4') and (rand[ind,i] <= p_matrix[4][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '4') and (rand[ind,i] >= p_matrix[4][0]): 
                    next_state = '5'
                    loss_pattern[ind,i] = 0
                elif (state == '5') and (rand[ind,i] <= p_matrix[5][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '5') and (rand[ind,i] >= p_matrix[5][0]): 
                    next_state = '6'
                    loss_pattern[ind,i] = 0
                elif (state == '6') and (rand[ind,i] <= p_matrix[6][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '6') and (rand[ind,i] >= p_matrix[6][0]): 
                    next_state = '7'
                    loss_pattern[ind,i] = 0
                elif (state == '7') and (rand[ind,i] <= p_matrix[7][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '7') and (rand[ind,i] >= p_matrix[7][0]): 
                    next_state = '7'
                    loss_pattern[ind,i] = 0
                state = next_state
        loss_pattern = th.stack(([loss_pattern]*11),dim = 2).cuda()
        return loss_pattern            
        
    prob_transition_matrix_light = [
        [0.8,0.2,0.0],
        [0.9,0.0,0.1],
        [0.9,0.0,0.1]
    ]   
    def generate_loss_pattern_light(self, episode_length = 200, p_matrix = prob_transition_matrix_light):
        rand = th.rand((episode_length,3))
        loss_pattern = th.zeros((episode_length,3))
        for i in range(3):
            state = '0'   # '0' means no error, '1' means error
            for ind in range(episode_length):
                if (state == '0') and (rand[ind,i] <= p_matrix[0][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 1
                elif (state == '0') and (rand[ind,i] >= p_matrix[0][0]):
                    next_state = '1'
                    loss_pattern[ind,i] = 1
                elif (state == '1') and (rand[ind,i] <= p_matrix[1][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '1') and (rand[ind,i] >= p_matrix[1][0]): 
                    next_state = '2'
                    loss_pattern[ind,i] = 0
                elif (state == '2') and (rand[ind,i] <= p_matrix[2][0]): 
                    next_state = '0'
                    loss_pattern[ind,i] = 0
                elif (state == '2') and (rand[ind,i] >= p_matrix[2][0]): 
                    next_state = '2'
                    loss_pattern[ind,i] = 0
                state = next_state
        loss_pattern = th.stack(([loss_pattern]*11),dim = 2).cuda()
        return loss_pattern          
        
    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        #######generate the loss pattern#############
        loss_pattern = self.generate_loss_pattern_heavy(episode_length = 500)
        counter = 0;
        #############################################
        
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            
            

            #############################################
            counter = counter + 1;
            #print(counter)
            #print(loss_pattern[counter].shape)
            actions = self.mac.select_actions(loss_pattern[counter], self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            ############# add this when evaluate our RL method
            #actions = actions[0]
            ##########################################
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(loss_pattern[counter], self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        ############# add this when evaluate our RL method
        #actions = actions[0]
        ##########################################        
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
