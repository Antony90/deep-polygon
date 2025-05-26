import datetime
from io import BytesIO
import json
import numpy as np
import requests
import requests_toolbelt.multipart.encoder as multipart_encoder

from PIL import Image
import matplotlib.pyplot as plt
import os

from agent import Agent
from env.state import GridState

from tqdm import tqdm

from train.stats import TrainingStats

class Webhook:
    def __init__(self, id: str, token: str, session_name: str, stats: TrainingStats, progress_bar: tqdm) -> None:
        self.url = f"https://discord.com/api/webhooks/{id}/{token}"
        self.session_name = session_name or str(datetime.datetime.now().date())

        response = requests.post(
            f'{self.url}?wait=true',
            data={
                "content": "Training Started",
                "thread_name": self.session_name
            }
        )
        response.raise_for_status()
        self.thread_id = json.loads(response.text)['channel_id']
        self.stats = stats
        self.pbar = progress_bar
        plt.style.use("dark_background")

    @staticmethod
    def field(name, value, inline=True):
        return {
            "name": name,
            "value": value,
            "inline": inline
        }

    @staticmethod
    def generate_gif(replay_history):
        frames = [GridState.to_img(state) for state in replay_history]
        extra_frames = [frames[-1].copy() for _ in range(10)] # make end state visible longer
        frames = frames + extra_frames

        frame_start = frames[0]
        gif_bytes = BytesIO()
        frame_start.save(
            gif_bytes,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            interlace=True,
            duration=80,
            loop=0
        )
        return gif_bytes

    @staticmethod
    def q_val_graph(q_vals):
        plt.plot(q_vals, color='green', linewidth=0.5)
        plt.xlabel('Action #')
        plt.ylabel('Q-value')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )
        img = BytesIO()
        plt.savefig(img, format='png', transparent=False)
        plt.clf()
        return img
    
    @staticmethod
    def loss_graph(losses):
        plt.plot(losses, color='red', linewidth=0.5)
        plt.xlabel('Training Steps since Last Mean')
        plt.ylabel('Loss')
        
        img = BytesIO()
        plt.savefig(img, format='png', transparent=False)
        plt.clf()
        return img
    
    @classmethod
    def ep_length_graph(cls, ep_lengths, mean_ep_lengths, mean_freq):
        return cls.generic_ep_mean_graph(
            y_data=ep_lengths,
            moving_avg_y_data=mean_ep_lengths,
            mean_freq=mean_freq,
            y_axis_label="Episode Length",
            y_data_color="blue",
            moving_avg_y_data_color="red"
        )

    @classmethod
    def reward_graph(cls, rewards, moving_avg_rewards, mean_freq):
        return cls.generic_ep_mean_graph(
            y_data=rewards,
            moving_avg_y_data=moving_avg_rewards,
            mean_freq=mean_freq,
            y_axis_label="Total Reward",
            y_data_color="white",
            moving_avg_y_data_color="red"
        )

    @staticmethod
    def generic_ep_mean_graph(
        y_data: list, 
        moving_avg_y_data: list, 
        mean_freq: int, 
        y_axis_label: str,
        y_data_color: str,
        moving_avg_y_data_color: str
    ) -> BytesIO:
        """Method for plotting statistics gathered each episode, with a moving average."""
        ep_nums = range(1, len(y_data)+1)
        ep_nums_interval = range(mean_freq, len(y_data)+1, mean_freq)
        
        plt.plot(ep_nums, y_data, linewidth=0.5, color=y_data_color)
        plt.plot(ep_nums_interval, moving_avg_y_data, linewidth=1.5, color=moving_avg_y_data_color)
        
        plt.ylabel(y_axis_label)
        plt.xlabel('Episode')
        # plt.tick_params(
        #     axis='x',
        #     which='both',
        #     bottom=False,
        #     top=False,
        #     labelbottom=False
        # )
        img = BytesIO()
        plt.savefig(img, format='png', transparent=False)
        plt.clf() # clear figure for next plot
        return img
    
    def send_replay_gif(self, replay_gif: BytesIO, new_best: int, length: int, rand_chance: float, kills: int, area: int):
        title = f"**New Best: {new_best:.4f}** ({kills} kills, {area:,} area, {length:,} actions, {rand_chance*100:.2f}% random)"
        body = {
            "content": title,
            "attachments": [
                {
                    "id": 0,
                    "filename": "run.gif",
                }
            ]
        }
        data = multipart_encoder.MultipartEncoder({
            "payload_json": json.dumps(body),
            "files[0]": ("run.gif", replay_gif, "image/gif")
        })
        requests.post(
            f'{self.url}?thread_id={self.thread_id}',
            data=data,
            headers={'Content-Type': data.content_type},
        )

    def send_report(
        self,
        best_reward: int,
        last_mean: int,
        train_ctr: int,
        num_train_steps: int,
        elapsed: int,
        remaining: int,
        ep_reward_graph: BytesIO,
        ep_length_graph: BytesIO,
        q_val_graph: BytesIO,
        loss_graph: BytesIO
    ):
        title = "Status Report"

        percent_done = (100*train_ctr/num_train_steps)
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed))
        if percent_done < 100:
            time_field = self.field("Time remaining (elapsed)", f"{str(datetime.timedelta(seconds=remaining))} ({elapsed_time_str})")
        else:
            time_field = self.field("Time elapsed", elapsed_time_str)

        images = [
            {"url": "attachment://reward.png"},
            {"url": "attachment://length.png"},
            {"url": "attachment://qvals.png"},
            {"url": "attachment://loss.png"}
        ]
        # generate duplicate embed with each image as an attachment
        # using the same URL for each embed will let images be combined into the same embed
        # much like sending multiple images in a normal discord message
        embeds = [
            {
                "title": f"{title} | {percent_done:.2f}%",
                "url": "https://www.google.com/", # any URL
                "color": 16760576,
                "fields": [
                    self.field("Best total reward", "{:.3f}".format(best_reward)),
                    self.field("Last mean reward", "{:.3f}".format(last_mean)),
                    time_field,
                    self.field("Training steps (total)", f"{train_ctr} ({num_train_steps})"),
                ],
                "image": image
            }
        for image in images]
        
        body = {"embeds": embeds}

        data = multipart_encoder.MultipartEncoder({
            "payload_json": json.dumps(body),
            "files[0]": ("reward.png", ep_reward_graph, "image/png"),
            "files[1]": ("length.png", ep_length_graph, "image/png"),
            "files[2]": ("qvals.png", q_val_graph, "image/png"),
            "files[3]": ("loss.png", loss_graph, "image/png")
        })
        
        response = requests.post(
            f'{self.url}?thread_id={self.thread_id}',
            data=data,
            headers={'Content-Type': data.content_type},
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print("Error sending embed")
            print(e)
            print(response.text)

    def send_msg(self, message: str, alert=False):
        if alert:
            message = '@here: ' + message
        data = { "content": message }
        response = requests.post(
            f'{self.url}?thread_id={self.thread_id}',
            data=data
        )
        response.raise_for_status()

    def post_new_best(self, best_run: list[np.ndarray], total_reward: int, rand_chance: float, kills: int, area: int):
        replay_gif = Webhook.generate_gif(best_run)
        length = len(best_run)
        self.send_replay_gif(replay_gif, total_reward, length, rand_chance, kills, area)
        replay_gif.close()


    def post_report(self):
        
        # generate graphs, store in memory as .pngs
        ep_reward_graph = self.reward_graph(self.stats.ep_rewards, self.stats.mean_ep_rewards, self.stats.mean_interval)
        ep_length_graph = self.ep_length_graph(self.stats.ep_lengths, self.stats.mean_ep_lengths, self.stats.mean_interval)
        q_val_graph = self.q_val_graph(self.stats.q_vals)
        loss_graph = self.loss_graph(self.stats.ep_avg_losses)
        
        elapsed = self.pbar.format_dict['elapsed']
        rate = self.pbar.format_dict['rate']
        remaining = (self.pbar.total - self.pbar.n) / rate if rate else 0
        
        remaining = int(remaining)
        elapsed = int(elapsed)

        # post report, close file descriptors
        self.send_report(
            train_ctr=self.pbar.n,
            best_reward=self.stats.best_reward,
            last_mean=self.stats.mean_ep_rewards[-1], 
            num_train_steps=self.pbar.total, 
            elapsed=elapsed, 
            remaining=remaining, 
            ep_reward_graph=ep_reward_graph,
            ep_length_graph=ep_length_graph,
            q_val_graph=q_val_graph,
            loss_graph=loss_graph
        )
        
        ep_reward_graph.close()
        ep_length_graph.close()
        q_val_graph.close()



    def post_webhook(self, new_best: bool, ep_rewards: list[int], best_run, best_reward, train_pbar, means, save_freq, agent, q_vals):
        # probably broken, use post_report or post_new_best
        if best_run is None or len(ep_rewards) == 0:
            return
        
        if new_best: # Only generate and send a gif
            replay_gif = Webhook.generate_gif(best_run)
            self.send_replay_gif(replay_gif, best_reward)
            replay_gif.close()
        else: # Send the report embed with graphs
            # Get time remaining and elapsed from progress bar
            elapsed = train_pbar.format_dict['elapsed']
            rate = train_pbar.format_dict['rate']
            remaining = (train_pbar.total - train_pbar.n) / \
                rate if rate else 0

            # Generate graphs in and store memory
            reward_graph = Webhook.reward_graph(ep_rewards, means, save_freq)
            q_val_graph = Webhook.q_val_graph(q_vals)
            self.send_report(agent.get_eps(), best_reward, means[-1], agent.train_ctr,
                             agent.num_train_steps, agent.replay.percent_full(), 0,
                             int(elapsed), int(remaining), reward_graph, q_val_graph)
            reward_graph.close()
            q_val_graph.close()
