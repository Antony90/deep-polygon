from typing import Optional
import statistics

from web.message import MeanStatistics
from web.websocket import WebSocketHandler


class TrainingStats:
    def __init__(self, ws_handler: WebSocketHandler, mean_freq: int = 1000):
        """
        Tracks episode-wise stats and computes moving averages over intervals.

        Args:
            mean_freq (int): Number of episodes per moving average interval.
        """
        self.ws_handler = ws_handler # to push statistic updates
        self.best_reward = -float("inf")
        self.ep_num = 0
        self.mean_interval = mean_freq

        # Raw episode stats
        self.ep_rewards = []
        self.ep_kills = []
        self.ep_land_captured = []
        self.ep_lengths = []
        self.ep_avg_losses = []

        # Moving averages over intervals
        self.mean_ep_rewards = []
        self.mean_ep_kills = []
        self.mean_ep_land_captured = []
        self.mean_ep_lengths = []
        self.mean_ep_avg_losses = []

        # Min/max/stddev within current interval (last mean_interval episodes)
        self.max_ep_length_interval: Optional[int] = None
        self.min_ep_length_interval: Optional[int] = None
        self.max_ep_avg_loss_interval: Optional[float] = None
        self.std_dev_ep_avg_loss_interval: Optional[float] = None

    def log_episode_stats(self, reward, length, avg_loss, kills, land_captured):
        self.ep_num += 1

        self.ep_rewards.append(reward)
        self.ep_lengths.append(length)
        self.ep_avg_losses.append(avg_loss)
        self.ep_kills.append(kills)
        self.ep_land_captured.append(land_captured)

        if reward > self.best_reward:
            self.best_reward = reward
            # Update leaderboard

        if self.ep_num % self.mean_interval == 0:
            # Last interval slice
            start_idx = max(0, self.ep_num - self.mean_interval)
            self.calc_moving_avg(start_idx, self.ep_num)
            stat_payload = self.generate_mean_stats_payload(start_idx, self.ep_num)
            self.ws_handler.put_broadcast_payload(stat_payload)

    def calc_moving_avg(self, start_idx, end_idx):
        interval_rewards = self.ep_rewards[start_idx : end_idx]
        interval_lengths = self.ep_lengths[start_idx : end_idx]
        interval_losses = self.ep_avg_losses[start_idx : end_idx]
        interval_kills = self.ep_kills[start_idx : end_idx]
        interval_land = self.ep_land_captured[start_idx : end_idx]

        # Calculate means
        self.mean_ep_rewards.append(sum(interval_rewards) / len(interval_rewards))
        self.mean_ep_lengths.append(sum(interval_lengths) / len(interval_lengths))
        self.mean_ep_avg_losses.append(sum(interval_losses) / len(interval_losses))
        self.mean_ep_kills.append(sum(interval_kills) / len(interval_kills))
        self.mean_ep_land_captured.append(sum(interval_land) / len(interval_land))

        # Min/max for episode length
        self.max_ep_length_interval = max(interval_lengths)
        self.min_ep_length_interval = min(interval_lengths)

        # Max and std dev for avg losses
        self.max_ep_avg_loss_interval = max(interval_losses)
        self.std_dev_ep_avg_loss_interval = (
            statistics.stdev(interval_losses) if len(interval_losses) > 1 else 0.0
        )

    def generate_mean_stats_payload(self, start_idx, end_idx):
        
        return MeanStatistics(
            avg_ep_reward=MeanStatistics.AvgEpisodeReward(
                mean=self.mean_ep_rewards[-1],                # latest
                values=self.ep_rewards[start_idx : end_idx],  # batch
                num_kills=self.mean_ep_kills[-1],             # latest
                land_captured=self.mean_ep_land_captured[-1], # latest
            ),
            avg_ep_length=MeanStatistics.AvgEpisodeLength(
                mean=self.mean_ep_lengths[-1],
                values=self.ep_lengths[start_idx : end_idx],
                max=self.max_ep_length_interval,
                min=self.min_ep_length_interval,
            ),
            avg_loss=MeanStatistics.AvgLoss(
                mean=self.mean_ep_avg_losses[-1],
                values=self.ep_avg_losses[start_idx : end_idx],
                max=self.max_ep_avg_loss_interval,
                std_dev=self.std_dev_ep_avg_loss_interval,
            ),
        )
