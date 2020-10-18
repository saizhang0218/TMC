import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import numpy as np
from torch.optim import RMSprop

# learning for 6h_vs_8z scenario
class QLearner_6h_vs_8z:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.params += list(self.mac.msg_rnn.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.loss_weight = [0.5,1,1.5]   # this is the beta in the Algorithm 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        ################ store the previous mac################
        previous_msg_list = []
        smooth_loss_list = []
        regulariation_smooth = 3.0
        regulariation_robust = 0.08
        self.mac.init_hidden(batch.batch_size)
        smooth_loss = th.zeros((192)).cuda()
        for t in range(batch.max_seq_length):
            agent_local_outputs, input_hidden_states, vi = self.mac.forward(batch, t=t)  
            input_hidden_states = input_hidden_states.view(-1,64)
            self.mac.hidden_states_msg, dummy = self.mac.msg_rnn(self.mac.hidden_states_msg, input_hidden_states)
            ss = min(len(previous_msg_list),3)
            #compute the l2 distance in the window
            for i in range(ss):
                smooth_loss += self.loss_weight[i] * (((dummy-previous_msg_list[i])**2).sum(dim = 1))/((ss*32*6*14*(dummy**2)).sum(dim = 1))
            previous_msg_list.append(dummy)
            if(len(previous_msg_list)>3): previous_msg_list.pop(0)      
            smooth_loss_reshape = smooth_loss.reshape(32,6,1).sum(1)   #(32,1)
            smooth_loss_list.append(smooth_loss_reshape)
            # generate the message
            dummy_final = dummy.reshape(32,6,14)
            dummy0 = dummy_final[:,0,:]
            dummy1 = dummy_final[:,1,:]
            dummy2 = dummy_final[:,2,:]       
            dummy3 = dummy_final[:,3,:]
            dummy4 = dummy_final[:,4,:]
            dummy5 = dummy_final[:,5,:]    
        
            agent0 = (dummy1 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            agent1 = (dummy0 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            agent2 = (dummy0 + dummy1 + dummy3 + dummy4 + dummy5)/5.0
            agent3 = (dummy0 + dummy1 + dummy2 + dummy4 + dummy5)/5.0
            agent4 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy5)/5.0
            agent5 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy4)/5.0
            agent_global_outputs =th.cat((agent0.view((32,1,14)),agent1.view((32,1,14)),agent2.view((32,1,14)),agent3.view((32,1,14)),agent4.view((32,1,14)),agent5.view((32,1,14))),1)            
            agent_outs = agent_local_outputs + agent_global_outputs
            mac_out.append(agent_outs)
        
        mac_out = th.stack(mac_out, dim=1)  # Concat over time   #(32,T,6,14)
        ############compute the robustness loss##################
        robust_loss = th.topk(mac_out,2)[0][:,:,:,0]-th.topk(mac_out,2)[0][:,:,:,1]
        robust_loss = th.exp(-25.0 * robust_loss).sum(dim = 2)[:, :-1].unsqueeze(2)/(32*6)    #(32,38)
        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_local_outputs, target_input_hidden_states, tvi = self.target_mac.forward(batch, t=t)
            target_input_hidden_states = target_input_hidden_states.view(-1,64)
            self.target_mac.hidden_states_msg, target_dummy = self.target_mac.msg_rnn(self.target_mac.hidden_states_msg, target_input_hidden_states)
            
            target_dummy = target_dummy.reshape(32,6,14)
            dummy0 = target_dummy[:,0,:]
            dummy1 = target_dummy[:,1,:]
            dummy2 = target_dummy[:,2,:]       
            dummy3 = target_dummy[:,3,:]
            dummy4 = target_dummy[:,4,:]
            dummy5 = target_dummy[:,5,:]   
            
            target_agent0 = (dummy1 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            target_agent1 = (dummy0 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            target_agent2 = (dummy0 + dummy1 + dummy3 + dummy4 + dummy5)/5.0
            target_agent3 = (dummy0 + dummy1 + dummy2 + dummy4 + dummy5)/5.0
            target_agent4 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy5)/5.0
            target_agent5 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy4)/5.0  
            
            target_agent_global_outputs = th.cat((target_agent0.view((32,1,14)),target_agent1.view((32,1,14)),target_agent2.view((32,1,14)),target_agent3.view((32,1,14)),target_agent4.view((32,1,14)),target_agent5.view((32,1,14))),1)
            target_agent_outs = target_agent_local_outputs + target_agent_global_outputs
            target_mac_out.append(target_agent_outs)
          
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out[avail_actions == 0] = -9999999
            cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())   #(32,25,1)
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        ######compute the smooth_loss and robust_loss#########
        smooth_loss = th.stack(smooth_loss_list[0:-1],dim = 1)
        smooth_loss = (smooth_loss * mask).sum()/mask.sum()
        robust_loss = (robust_loss * mask).sum()/mask.sum()

        loss = (masked_td_error ** 2).sum() / mask.sum() + regulariation_smooth * smooth_loss + regulariation_robust * robust_loss
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


