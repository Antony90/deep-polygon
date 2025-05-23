import base64
from collections import defaultdict
from io import BytesIO
from queue import Queue
import random
from threading import Thread
from typing import Optional

import colorama
from tqdm import tqdm

from agent import Agent
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

        self.player_to_state = {}
        self.player_ep_length = defaultdict(int)

    def get_state(self, player_id):
        return self.player_to_state[player_id]

    def set_state(self, player_id: int, state):
        self.player_to_state[player_id] = state

    def inc_ep_length(self, player_id: int):
        self.player_ep_length[player_id] += 1

    def pop_ep_length(self, player_id: int):
        return self.player_ep_length.pop(player_id)

    def get_players(self):
        return [player for player in self.player_to_state]


class RenderManager:
    """
    Tracks the highest reward alive player
    Stores a history of the new best replays
    """

    def __init__(self):
        self.player_id = None
        self.state_queue: Queue[tuple[bytes, float]] = Queue()

    def get_best_score_replay(self) -> list[GridState]:
        return

    def set_spectate_player(self, player_id):
        # TODO: validation on `env` to check if player exists
        self.player_id = player_id
        
    def queue_state(self, state: any, reward):
        grid_state, _ = state
        img = GridState.to_img(grid_state)
        # TODO: optimize packet size
        buffer = BytesIO()
        img.save(buffer, format="png")
        img_b64 = base64.b64encode(img)
        # Submit for websocket server to push to client
        self.state_queue.put((img_b64, reward))

    def pop_state(self):
        return self.state_queue.get()
    
    def empty(self):
        return self.state_queue.qsize() == 0


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
        self._spawn_players()
        self._run_train_loop()

    def stop(self):
        self.run = False

    def _init_scheduler(self):
        if self.webhook:
            self.scheduler.add_job(
                self.webhook, "interval", minutes=30, max_instances=1
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

    def _spawn_players(self):
        # Spawn correct player type for each agent
        for group in self.player_groups:
            for _ in range(group.num_players):
                player, initial_state = self.env.spawn_player(group.player_cls)

                group.set_state(player, initial_state)

    def _run_train_loop(self):
        self.run = True
        while self.run:
            for player_group in self.player_groups:
                agent = player_group.agent
                players = player_group.get_players()

                for player in players:
                    state = player_group.get_state(player)

                    action, q_val = agent.act(state)
                    new_state, reward, done = self.env.step(player, action)

                    agent.replay.store(state, action, reward, new_state, done)

                    loss = agent.train()

                    # Update Telemetry
                    self.stats.add_q_val(q_val)
                    self.stats.add_loss(loss)
                    player_group.inc_ep_length(player)

                    # Rendering logic
                    if self.render_manager.player_id == player.id:
                        self.render_manager.queue_state((new_state, reward))

                    if done:
                        ep_length = player_group.pop_ep_length(player)
                        self.stats.update_episode(reward, ep_length)
                        
                        player, new_state = self.env.spawn_player(
                            player_group.player_cls, player.id
                        )

                    player_group.set_state(player, new_state)

                # Increment progress bar manually
                self.pbar.update(len(players))
