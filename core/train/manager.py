import base64
from collections import defaultdict
from io import BytesIO
import random
from threading import Thread
from asyncio import Queue as AsyncQueue
from queue import Queue as BlockingQueue
from typing import Optional

import colorama
import numpy as np
import torch
from tqdm import tqdm

from agent import Agent
from constants import STATE_SHAPE
from train.render import RenderManager
from env.game import Game
from env.player.base import Player
from apscheduler.schedulers.background import BackgroundScheduler

from env.state import GridState
from train.stats import TrainingStats
from web.message import LiveFrameData, TrainingProgress
from web.webhook import Webhook
from web.websocket import WebSocketHandler


class AgentPlayerGroup:
    """
    Represents the 1-to-Many relationship of
    1 Agent to many players, where Players are all of the same class
    and aren't initially instantiated.
    All players share the same Agent AI, however this does NOT
    make the agents collaborative/friendly to each other.
    There is a method to spawn all the players in a given env

    Also serves to hold per-agent episode stats for each player
    e.g. episode length, total reward, kills, land captured
    """

    def __init__(self, agent: Agent, player_cls: Player, num_players: int, torch_device: str):
        self.agent = agent
        self.player_cls = player_cls
        self.num_players = num_players

        self.player_to_state: dict[int, any] = {}
        self.players: dict[int, Player] = {}
        
        # Per episode stats
        self._ep_length = defaultdict(int)
        self._ep_loss = defaultdict(lambda: torch.tensor(0.0).to(torch_device))
        self._ep_total_reward = defaultdict(float)

    def spawn_player(self, env: Game, replace_player_id: Optional[int] = None):
        player, new_state = env.spawn_player(self.player_cls, replace_player_id)
        self.players[player.id] = player

        return player, new_state

    def get_players(self):
        return [player for player in self.players.values()]

    def get_state(self, player_id):
        return self.player_to_state[player_id]

    def set_state(self, player_id: int, state):
        self.player_to_state[player_id] = state

        
    def add_step_stats(self, player_id: int, reward: float, loss_tensor: Optional[torch.Tensor]):
        # Accumulate raw loss tensor (on same device as loss_tensor)
        if loss_tensor is not None:
            self._ep_loss[player_id] = self._ep_loss[player_id] + loss_tensor.detach()
        self._ep_total_reward[player_id] += reward
        self._ep_length[player_id] += 1
        

    def get_ep_length(self, player_id: int):
        return self._ep_length[player_id]
    
    def get_ep_total_reward(self, player_id: int):
        return self._ep_total_reward[player_id]

    def pop_ep_stats(self, player_id):
        if self._ep_length[player_id] == 0:
            avg_loss = 0.0
        else:
            avg_loss = (self._ep_loss[player_id] / self._ep_length[player_id]).item()
            
        total_reward = self._ep_total_reward[player_id]
        ep_length = self._ep_length[player_id]
        
        # Reset accumulators, let defaultdict reinitialize
        del self._ep_loss[player_id]
        del self._ep_length[player_id]
        del self._ep_total_reward[player_id]
        
        return total_reward, ep_length, avg_loss


class TrainingManager:
    def __init__(
        self,
        map_size: int,
        player_group: list[AgentPlayerGroup],
        stats: TrainingStats,
        progress_bar: tqdm,
        render_manager: RenderManager,
        ws_handler: WebSocketHandler,
        webhook: Optional[Webhook] = None,
    ):

        self.env = Game(map_size=map_size)
        self.player_groups = player_group
        self.stats = stats
        self.render_manager = render_manager
        self.ws_handler = ws_handler
        self.webhook = webhook

        self.scheduler = BackgroundScheduler()
        self.pbar = progress_bar

        self.run = False

    def start(self):
        self._init_scheduler()
        self._spawn_initial_players()
        self._run_train_loop()

    def stop(self):
        self.run = False

    def _init_scheduler(self):
        if self.webhook:
            self.scheduler.add_job(
                self.webhook.post_report, "interval", minutes=30, max_instances=1
            )

        # We'll track the first AI on the progress bar for now
        tracking_agent = self.player_groups[0].agent

        def update_pbar():
            self.pbar.set_postfix(
                {
                    "eps": tracking_agent.get_eps(),
                    "episode": self.stats.ep_num,
                    f"mean_reward": self.stats.mean_ep_rewards[-1] if len(self.stats.mean_ep_rewards) else None,
                    "mean_loss": self.stats.mean_ep_avg_losses[-1] if len(self.stats.mean_ep_avg_losses) else None,
                    "max_reward": self.stats.best_reward,
                }
            )
            progress = TrainingProgress.generate_payload(self.pbar, tracking_agent)
            self.ws_handler.put_broadcast_payload(progress)

        self.scheduler.add_job(update_pbar, "interval", seconds=5, max_instances=1)
        self.scheduler.start()

    def _spawn_initial_players(self):
        # Spawn correct player type for each agent
        for group in self.player_groups:
            for _ in range(group.num_players):
                player, initial_state = group.spawn_player(self.env)

                group.set_state(player.id, initial_state)

    def _run_train_loop(self):
        self.run = True
        while self.run:
            for player_group in self.player_groups:
                agent = player_group.agent
                players = player_group.get_players()

                for player in players:
                    state = player_group.get_state(player.id)

                    action, q_val = agent.act(state)
                    new_state, reward, done = self.env.step(player, action)

                    agent.replay.store(state, action, reward, new_state, done)

                    loss = agent.train()

                    # Update Telemetry
                    player_group.add_step_stats(player.id, reward, loss)

                    # Rendering logic
                    if self.render_manager.player_id == player.id:
                        total_reward = player_group.get_ep_total_reward(player.id)
                        ep_length = player_group.get_ep_length(player.id)
                        
                        frame_data = LiveFrameData(
                            reward=reward,
                            total_reward=total_reward,
                            ep_length=ep_length,
                            kills=player.kills,
                            land_captured=player.area,
                            rank=0 
                            # TODO: impl efficient player ranking and leaderboard tracking
                            # TODO: make this a Partial class and have rank, image processed async with render mgr
                        )
                        self.render_manager.queue_state(new_state, frame_data)

                    if done:
                        self.stats.log_episode_stats(*player_group.pop_ep_stats(player.id), player.kills, player.area)

                        player, new_state = player_group.spawn_player(
                            self.env, player.id
                        )

                    player_group.set_state(player.id, new_state)

                # Increment progress bar manually
                self.pbar.update(len(players))
