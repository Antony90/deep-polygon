import json
import os
import random
import time
from typing import Callable, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from constants import STATE_N_VARS, STATE_SHAPE_CHANNELS_FIRST, Direction
from model import GridNet
from replay import MultiInputReplayBuffer, ReplayBuffer
from colorama import Fore, Style

StateType = tuple[np.ndarray, np.ndarray]

class Agent:
    model_path = "./models"

    def __init__(
        self, 
        batch_size: int, 
        gamma: float, 
        learning_rate: float, 
        eps_decay_steps: int, 
        update_target_rate: int,
        epsilon_min: float,
        device: torch.device,
        replay: ReplayBuffer,
        num_actions: int,
        policy_net: nn.Module,
        target_net: nn.Module,
        eval,
        model_path : Optional[str] = None,
        load_model : Optional[str] = None
    ):

        self.num_actions = num_actions

        # hyperparameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.eps_decay_steps = eps_decay_steps
        self.update_target_rate = update_target_rate
        self.train_cntr = 0
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min

        self.replay = replay

        self.device = device
        if model_path is None:
            if not os.path.exists(self.model_path):
                print(f"Created default model directory at: {self.model_path}")
                os.mkdir("./models")
        else:
            self.model_path = model_path
            if not os.path.exists(self.model_path):
                raise Exception(f"Supplied model directory `{self.model_path}` does not exist.")

        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss().to(device)

        self.clip_grad = 80.0 # clip gradient value
        self.eval = eval # inference only mode

        if load_model:
            self.load(load_model, verbose=True)
            
        if self.eval:
            if load_model is None:
                raise Exception("Must load a model in evaluation mode.")
            else:
                self.load(load_model)
                self.eval_msg(load_model)
        else:
            self.summary_msg()

        print()

    def eval_msg(self, model_name: str):
        print()
        print(Style.BRIGHT + Fore.GREEN + f"Inference only mode! All actions use the loaded model: {model_name}" + Style.RESET_ALL)

    def summary_msg(self):
        print()
        print(Style.BRIGHT + Fore.GREEN + "Hyperparameters" + Style.NORMAL)
        print(f"\tbatch_size = {self.batch_size}")
        print(f"\tlearning_rate = {self.learning_rate}")
        print(f"\tgamma = {self.gamma}")
        print(f"\tmemory_size = {self.replay.size:,}")
        print(f"\ttrain_steps = {self.eps_decay_steps:,}")
        print(f"\ttarget_update = {self.update_target_rate:,}")

        total_params = 0
        for parameter in self.policy_net.parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            total_params+=params

        print(Style.BRIGHT + Fore.WHITE + "Other Parameters" + Style.NORMAL)
        print(f"\ttotal_model_parameters = {total_params:,}")
        print(f"\tepsilon_min = {self.epsilon_min}")
        print(f"\tdevice = {self.device}")
        print(f"\tmodel_path = {self.model_path}")
        print(f"\toptimizer = {self.optimizer.__class__.__name__}")
        print(f"\tloss_fn = {self.loss_fn.__class__.__name__}")
        print()
        print(f"[Ctrl+C to exit]" + Style.RESET_ALL)

    @classmethod
    def from_config(cls, config_path: str, load_model: Optional[str] = None, eval=False):
        with open(config_path, "r") as f:
            config = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = GridNet(num_channels=STATE_SHAPE_CHANNELS_FIRST[0], num_scalars=STATE_N_VARS, num_actions=len(Direction)).to(device)
        target_net = GridNet(num_channels=STATE_SHAPE_CHANNELS_FIRST[0], num_scalars=STATE_N_VARS, num_actions=len(Direction)).to(device)
     
        replay = MultiInputReplayBuffer(
            size=config.pop("memory_size"),
            grid_shape=STATE_SHAPE_CHANNELS_FIRST, 
            num_scalars=STATE_N_VARS
        )

        return Agent(
            replay=replay,
            policy_net=policy_net,
            target_net=target_net,
            device=device,
            num_actions=policy_net.num_actions,
            eval=eval,
            load_model=load_model,
            **config
        )
    
    @classmethod
    def eval_pipeline(cls, model_name: str) -> Callable[[StateType], Direction]:
        """Load a model, returns the max Q value function.
        
        This function takes the state as a tuple `(grid, scalar)` and returns the max Q action, an integer."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = GridNet(num_channels=STATE_SHAPE_CHANNELS_FIRST[0], num_scalars=STATE_N_VARS, num_actions=len(Direction)).to(device)
    
        checkpoint = torch.load(os.path.join(cls.model_path, model_name))
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
    
        def act(state):
            state_tensor = cls.game_state_to_tensor_batch(state, device)
            q_values = policy_net.forward(state_tensor)[0]
            action = torch.argmax(q_values)
            
            return action.item()

        return act

    def update_target_network(self):
        # copy the weights of the target model to the current model
        # SOFT update
        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        # self.target_net.load_state_dict(target_net_state_dict)
        
        # HARD update
        self.target_net.load_state_dict(self.policy_net.state_dict(), strict=True)

    @staticmethod
    def game_state_to_tensor_batch(state, device):
        grid, scalar = state
        with torch.no_grad():
            grid_t = torch.tensor(grid[np.newaxis, :], device=device)
            scalar_t = torch.tensor(scalar[np.newaxis, :], device=device, dtype=torch.float32)
        return grid_t, scalar_t


    def get_eps(self):
        return max(1 - (self.train_cntr/self.eps_decay_steps), self.epsilon_min)
        
    def set_eps(self, value: float):
        """Set train counter such that `epsilon decay steps * epsilon = train counter`
        
        Returns new train counter value for progress bars."""
        if value > 1.0 or value < 0.0:
            raise Exception("Expected epsilon value in interval [0, 1]")

        self.train_cntr = int(value * self.eps_decay_steps)
        return self.train_cntr

    def act(self, state, is_grid=True):
        """Predicts the max Q action of a game state.
        Uses epsilon greedy strategy for exploration.
        
        In evaluation mode, all actions use the policy net. See `self.eval()`.
        
        Returns tuple of `[action, q_value]`."""

        if np.random.rand() <= self.get_eps() and not self.eval:
            # random action has no q-value
            return random.randrange(0, self.num_actions), None
        else:
            # convert to batch tensor 
            if is_grid:
                state_t = self.game_state_to_tensor_batch(state, self.device)
            else:
                state_t = torch.tensor(state[np.newaxis, :], device=self.device)
                
            # return action index and its q-value
            pred = self.policy_net.forward(state_t)[0]
            idx = torch.argmax(pred)
            return idx.item(), float(pred[idx])

    def train(self):
        if self.replay.cntr < self.batch_size:
            return float('inf')
        state, action, reward, new_state, done = self.replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            q_next = self.policy_net.forward(new_state)
            max_q_actions = q_next.argmax(1)
            q_next_target = self.target_net.forward(new_state)
            max_q_values = q_next_target.gather(1, max_q_actions.unsqueeze(1)).squeeze(1)

        q_eval = self.policy_net.forward(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # zero the future reward if terminal state
        q_target = reward + self.gamma * max_q_values * (1 - done)
        loss = self.loss_fn(q_eval, q_target)

        # start ORIGINAL
        # with T.no_grad():
        #     q_next = self.policy_net.forward(new_state)
        #     max_q_actions = q_next.argmax(1)
        #     q_next_target = self.target_net.forward(new_state)
        #     max_q_values = q_next_target[:, max_q_actions]

        # q_eval = self.policy_net.forward(state)[:, action]
        # # zero the future reward if terminal state
        # q_target = reward + self.gamma * max_q_values * (1 - done)
        
        # loss = self.loss_fn(q_eval, q_target)
        # end ORIGINAL

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.policy_net.parameters(), self.clip_grad)
        self.optimizer.step()

        self.train_cntr += 1
            
        if self.train_cntr % self.update_target_rate == 0:
            self.update_target_network()

        return loss.item()

    def train_steps_completed(self):
        """No. times `self.train()` has been called."""
        return self.train_cntr 

    
    def save(self, name, verbose=False):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(self.model_path, name))
        if verbose:
            print(Style.BRIGHT + Fore.GREEN + f"Saved model {name}" + Style.RESET_ALL)

    def load(self, name, verbose=False):
        checkpoint = torch.load(os.path.join(self.model_path, name))
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if verbose:
            print(Style.BRIGHT + Fore.GREEN + f"Loaded model {name}" + Style.RESET_ALL)
            
    @classmethod
    def convert_train_model(cls, model_name: str):
        """Opens a pickled trained model dict with keys for policy network, target network and the optimizer.
        
        Creates a new piclked model as the policy network's state dict.
        
        Must be called before a model can be used with `online` command.
        Must be called with cuda enabled since tensors from trained model
        expect to be loaded on a cuda device"""
        print(Fore.YELLOW + f"Warning: requires CUDA enabled device to open trained model!" + Fore.RESET)
        # load trained model and move to cpu
        trained_model = torch.load(os.path.join(cls.model_path, model_name))
        print(Fore.GREEN + f"Opened trained model {model_name}" + Fore.RESET)
        temp_net = GridNet(num_channels=STATE_SHAPE_CHANNELS_FIRST[0], num_scalars=STATE_N_VARS, num_actions=len(Direction))
        temp_net.load_state_dict(trained_model["policy_state_dict"])
        temp_net.to("cpu")
        print(Fore.GREEN + f"Moved tensors to CPU" + Fore.RESET)

        # only save policy state dict for inference
        new_model_name = model_name.split(".pt")[0] + "-eval.pt"
        torch.save(temp_net.state_dict(), os.path.join(cls.model_path, new_model_name))

        print(Fore.GREEN + f"Saved new model {new_model_name}" + Fore.RESET)



    def eval(self):
        """Enables evaluation only mode."""
        self.eval = True