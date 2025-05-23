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
from tqdm import tqdm

from agent import Agent
from constants import STATE_SHAPE
from train.render import RenderManager
from env.game import Game
from env.player.base import Player
from apscheduler.schedulers.background import BackgroundScheduler

from env.state import GridState
from train.stats import TrainingStats
from web.webhook import Webhook


# Represents the 1-to-Many relationship of
# 1 Agent to many players, where Players are all of the same class
# and aren't initially instantiated.
# All players share the same Agent AI, however this does NOT
# make the agents collaborative/friendly to each other.
# There is a method to spawn all the players in a given env
class AgentPlayerGroup:
    def __init__(self, agent: Agent, player_cls: Player, num_players: int):
        self.agent = agent
        self.player_cls = player_cls
        self.num_players = num_players

        self.player_to_state: dict[int, any] = {}
        self.player_ep_length = defaultdict(int)
        self.players: dict[int, Player] = {}
    
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

    def inc_ep_length(self, player_id: int):
        self.player_ep_length[player_id] += 1

    def pop_ep_length(self, player_id: int):
        return self.player_ep_length.pop(player_id)






class TrainingManager:
    def __init__(
        self,
        map_size: int,
        player_group: list[AgentPlayerGroup],
        stats: TrainingStats,
        progress_bar: tqdm,
        render_manager: RenderManager,
        webhook: Optional[Webhook] = None,
    ):

        self.env = Game(map_size=map_size)
        self.webhook = webhook
        self.player_groups = player_group
        self.render_manager = render_manager
        self.stats = stats

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
                    "max_r": self.stats.best_reward,
                    f"mean_r": self.stats.last_mean_ep_reward,
                    "episode": self.stats.ep_num,
                    "mean_loss": self.stats.mean_loss,
                }
            )

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
                    self.stats.add_q_val(q_val)
                    self.stats.add_loss(loss)
                    player_group.inc_ep_length(player.id)

                    # Rendering logic
                    if self.render_manager.player_id == player.id:
                        self.render_manager.queue_state(new_state, reward)

                    if done:
                        ep_length = player_group.pop_ep_length(player.id)
                        self.stats.update_episode(reward, ep_length)
                        
                        player, new_state = player_group.spawn_player(self.env, player.id)

                    player_group.set_state(player.id, new_state)

                # Increment progress bar manually
                self.pbar.update(len(players))
