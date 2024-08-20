import asyncio
import json
import sys
import os
import time
import random
import art
import numpy as np

import colorama
from colorama import Fore, Style
import torch
from tqdm import tqdm
from collections import deque
from threading import Event, Thread
from queue import Queue, Empty
from typing import Callable, Optional
from apscheduler.schedulers.background import BackgroundScheduler
import websockets as ws
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from webhook import Webhook
from constants import STATE_SHAPE, Direction
from agent import Agent, StateType

from env.state import Block, Overlay
from env.game import Game


def red(string: str):
    return Fore.RED + string + Fore.RESET

def yellow(string: str):
    return Fore.YELLOW + string + Fore.RESET

def green(string: str):
    return Fore.GREEN + string + Fore.RESET

def bold(string: str):
    return Style.BRIGHT + string + Style.RESET_ALL

class Playground:
    def __init__(self):
        self.envs: list[Game] = []
        self.should_render = False

    def create_env(self, *args, **kwargs):
        env = Game(*args, **kwargs)
        self.envs.append(env)
        return env
    

    def run_builder_server(self, agent: Agent, webhook: Optional[Webhook], rand_chance: Optional[float], num_builders: int, map_size: int):        
        if rand_chance:
            # no. train steps assumed to be done given this epsilon
            steps_done = agent.set_eps(rand_chance)
            num_steps = agent.eps_decay_steps - steps_done
        else:
            num_steps = agent.eps_decay_steps
        
        # progress towards reaching minimum epsilon value (random action chance)
        pbar = tqdm(total=num_steps, leave=False)

        # logging & counter vars
        best_reward = -float('inf')
        best_run = None # history of states for episode with best reward
        ep_num = 0
        mean_freq = 4_000 # save model, calculate a moving avg, in episodes
        
        ep_rewards = [] # rewards for every episode 
        mean_ep_rewards = [] # moving average, window size of `mean_freq`
        last_mean_ep_reward = None # last mean episode reward
        
        ep_lengths = [] # length of each episode
        mean_ep_lengths = [] # moving avg mean episode length
        last_mean_ep_length = None # last mean episode length

        q_vals = [] # q value of policy actions
        losses = [] # losses since last save interval
        mean_loss = None

        save_freq = 500_000 # interval in no. train steps
        
        # job to post webhook with training reports
        scheduler = BackgroundScheduler()
        if webhook:
            def post_wh():
                remaining, elapsed = webhook.calc_remaining_elapsed_from_pbar(pbar)
                webhook.post_report(
                    best_reward=best_reward,
                    mean_freq=mean_freq,
                    ep_rewards=ep_rewards,
                    mean_ep_rewards=mean_ep_rewards,
                    ep_lengths=ep_lengths,
                    mean_ep_lengths=mean_ep_lengths,
                    q_vals=q_vals,
                    agent=agent,
                    remaining=remaining,
                    elapsed=elapsed
                )

            interval = {}
            if agent.eps_decay_steps < 2_000_000:
                interval["minutes"] = 30
            elif agent.eps_decay_steps < 10_000_000:
                interval["hours"] = 1
            else:
                interval["hours"] = 2

            scheduler.add_job(post_wh, 'interval', max_instances=1, **interval)

        # update terminal progress bar
        def update_pbar():
            pbar.set_postfix({
                'eps': "{:.2f}".format(agent.get_eps()), 
                'maxr': "{:.3f}".format(best_reward), 
                f'mean_{mean_freq}r': "{:.3f}".format(last_mean_ep_reward) if last_mean_ep_reward is not None else last_mean_ep_reward, 
                'episode': ep_num, 
                'replay_mem%': "{:.2f}".format(agent.replay.percent_full()),
                'loss': mean_loss
            })

        scheduler.add_job(update_pbar, 'interval', seconds=5, max_instances=1)
        scheduler.start()
        
        env = self.create_env(map_size=map_size)

        # create render thread to empty queue
        next_state = Queue() # states to render
        render_flag = Event() # toggled when user presses Enter
        render_thread = Thread(target=self.render, args=(next_state, render_flag), daemon=True)
        render_thread.start()

        # listen for Enter key presses
        listen_stdin_thread = Thread(target=self.listen_stdin, args=(next_state, render_flag), daemon=True)
        listen_stdin_thread.start()

        watch = random.randrange(num_builders) # pick a Builder to watch
        
        if self.should_render:
            render_flag.set()
            print(colorama.ansi.clear_screen() + colorama.Cursor.POS(), end="")


        states = [env.spawn_builder() for _ in range(num_builders)] # spawn and take initial state
        agent_rewards = [0 for _ in range(num_builders)] # total reward for current episode
        ep_history = [[states[i][0]] for i in range(num_builders)] # history of states for current episode

        while True:
            for i, player in enumerate(env.players):
                # select an action
                action, q_val = agent.act(states[i])
                q_vals.append(q_val)

                # invoke action on env
                new_state, reward, done = env.step(player, action)

                # store transition and perform one train step
                agent.replay.store(states[i], action, reward, new_state, done)
                loss = agent.train()

                losses.append(loss)
                ep_history[i].append(new_state[0])
                agent_rewards[i] += reward

                if self.should_render and watch == i:
                    next_state.put((new_state[0], new_state[1], reward, agent_rewards[i], done))

                if agent.train_steps_completed() % save_freq == 0 and last_mean_ep_reward:
                    agent.save(f"{agent.train_steps_completed():08}_m{last_mean_ep_reward:5.3f}.pt")

                if done:
                    ep_num += 1
                    ep_rewards.append(agent_rewards[i])
                    ep_lengths.append(len(ep_history[i]))

                    if agent_rewards[i] > best_reward:
                        best_reward = agent_rewards[i]
                        best_run = ep_history[i]

                        # only post after initial influx of new bests
                        if agent.get_eps() < 0.9 and webhook:
                            webhook.post_new_best(best_run, best_reward, agent.get_eps(), player.kills, player.area)

                    # calculate mean and send status report with webhook
                    if ep_num % mean_freq == 0:
                        last_mean_ep_reward = sum(ep_rewards[-mean_freq:]) / mean_freq
                        mean_ep_rewards.append(last_mean_ep_reward)

                        last_mean_ep_length = sum(ep_lengths[-mean_freq:]) / mean_freq
                        mean_ep_lengths.append(last_mean_ep_length)

                        mean_loss = sum(losses) / mean_freq
                        losses.clear()
                    

                    # reset agent
                    agent_rewards[i] = 0 
                    ep_history[i].clear()
                    new_state = env.spawn_builder(index=i) # replace player and set starting state

                states[i] = new_state
                
            pbar.update(num_builders)


    def run_eval_model(self, model_name: str, agent: Agent, map_size: int):
        env = self.create_env(map_size=map_size)
        state = grid, scalar = env.spawn_builder()
        player = env.players[0]

        reward = action = total_reward = 0 # initial
        os.system("clear")
        
        while True:
            done = False
            while not done:
                # print last state, reward, action
                state_str = Playground.state_to_str(grid.transpose((1, 2, 0)), scalar, reward, total_reward)
                print(colorama.Cursor.POS(1, 1), end="")
                print(state_str)

                action, _ = agent.act(state)
                
                new_state, reward, done = env.step(player, action)
                total_reward += reward

                state = grid, scalar = new_state
    
            state = env.spawn_builder(index=0)
            player = env.players[0]
            total_reward = 0


    def listen_stdin(self, next_state: Queue, render_flag: Event):
        while True:
            for _ in sys.stdin:
                if render_flag.is_set():
                    render_flag.clear()
                    self.should_render = False
                    # clear accumulated frames since render is slower than incoming
                    # hacky way to empty a threadsafe object
                    while not next_state.empty():
                        try:
                            next_state.get(block=False)
                        except Empty:
                            continue
                        next_state.task_done()
                else:
                    render_flag.set()
                    self.should_render = True
                print(colorama.ansi.clear_screen() + colorama.Cursor.POS(), end="")

            time.sleep(5)

    def render(self, next_state: Queue, render_flag: Event):
        '''Consumer for states produced by an agent in `env`'''
        while True:
            render_flag.wait()

            grid, scalar, reward, total_reward, _ = next_state.get()
            grid = grid.transpose((1, 2, 0))
            print(Playground.state_to_str(grid, scalar, reward, total_reward))


    @staticmethod
    def state_to_str(grid: np.ndarray, scalar: np.ndarray, reward: float, total_reward: float) -> str:
        # TODO print scalar state too
        chrs = []
        chrs.append("┣"+"━" * STATE_SHAPE[1]+"┫\n")
        for i in range(STATE_SHAPE[0]):
            chrs.append("┃")
            for j in range(STATE_SHAPE[1]):
                [block, self_layer, enemy_layer] = grid[i, j]
                
                if self_layer != Overlay.EMPTY:
                    if self_layer == Overlay.trail_slot(0):
                        chrs.append(green("·"))
                    elif self_layer == Overlay.head_slot(0):
                        chrs.append(green("●"))
                    else:
                        chrs.append(green("?"))

                elif enemy_layer != Overlay.EMPTY:
                    if enemy_layer == Overlay.trail_slot(0):
                        chrs.append(red("·"))
                    elif enemy_layer == Overlay.head_slot(0):
                        chrs.append(red("●"))
                    elif enemy_layer == Overlay.trail_slot(1):
                        chrs.append(yellow("·"))
                    elif enemy_layer == Overlay.head_slot(1):
                        chrs.append(yellow("●"))
                    else:
                        chrs.append(red("?"))
                else:
                    if block == Block.EMPTY:
                        chrs.append(" ")
                    elif block == Block.SELF_BLOCK:
                        chrs.append(green("■"))
                    elif block == Block.WALL:
                        chrs.append(bold("▦"))
                    elif block == Block.enemy_block_for(0):
                        chrs.append(red("■"))
                    elif block == Block.enemy_block_for(1):
                        chrs.append(yellow("■"))
                    else:
                        chrs.append(bold("?"))
                            
            chrs.append("┃\n")
        chrs.append("┗"+"━" * STATE_SHAPE[1]+"┛")
        grid_str = ''.join(chrs)

        info_str = f"reward: {reward:.4f} ({total_reward:.4f})"
        pad_info = STATE_SHAPE[1] - len(info_str)
        pad_info_left = (pad_info // 2) * " "
        pad_info_right = (pad_info - (pad_info//2)) * " "

        scalar_str = str(scalar)
        pad_scalar = STATE_SHAPE[1] - len(scalar_str)
        pad_scalar_left = (pad_scalar // 2) * " "
        pad_scalar_right = (pad_scalar - (pad_scalar//2)) * " "


        return colorama.Cursor.POS(0, 0) + \
            "┃" + pad_info_left + bold(info_str) + pad_info_right + "┃\n" + \
            "┃" + pad_scalar_left + bold(scalar_str) + pad_scalar_right + "┃\n" + \
            grid_str

    def decode_user_input(key: str):
        match key:
            case 'w':
                return Direction.UP
            case 'a':
                return Direction.LEFT
            case 's':
                return Direction.DOWN
            case 'd':
                return Direction.RIGHT
            case 'p':
                return Direction.PAUSE
            case _:
                raise Exception(f"Unexpected key {key}")

    def run_user_input(self):
        env = self.create_env(max_players=None)
        grid, scalar = env.spawn_builder()
        reward = 0
        total_reward = 0

        while True:
            state_str = self.state_to_str(grid.transpose((1, 2, 0)), scalar, reward, total_reward)
            print(colorama.Cursor.POS(1, 1), end="")
            print(state_str)

            while True:
                try:
                    action = Playground.decode_user_input(input(">>> "))
                except Exception as e: print(e.args[0])
                else: break
            new_state, reward, done = env.step(env.players[0], action)
            total_reward += reward

            if done:
                print(Style.BRIGHT + Fore.RED + "You died, respawned" + Style.RESET_ALL)
                new_state = env.spawn_builder(0)
                total_reward = 0
                time.sleep(0.5)
                
            grid, scalar = new_state

    async def run_websocket_server(self, act_fn: Callable[[StateType], Direction]):
        def parse_state(state):
            grid = np.array(state["grid"])
            # 61x61 to 31x31
            # then convert to "channels first" image
            grid = grid[15:46, 15:46].transpose((2, 0, 1)).astype(np.float32)
            scalar = np.array(state["scalar"])
           
            r =  (grid, scalar)

            print(grid.shape)

            print(scalar.shape)
            print(scalar)
            return r

        async def handle_client(ws):
            try:
                async for message in ws:
                    state = json.loads(message)
                    state = grid, _ = parse_state(state)
                    render = self.state_to_str(grid.transpose((1, 2, 0)), scalar=np.ndarray([]), reward=0, total_reward=0)
                    print(render)
                    
                    action = act_fn(state)
                    await ws.send(json.dumps({ "action": action }))
            except ConnectionClosedError:
                print("bad disconnect")
            except ConnectionClosedOK:
                print("client closed ok")

        server_entrypoint = ws.serve(handle_client, port=4321) if os.environ.get('USE_TCP_SOCK') \
                    else ws.unix_serve(handle_client, path='/sock/controller.sock')
        async with server_entrypoint:
            await asyncio.Future()
    

def main(args):
    playground = Playground()
    if args.subcommand == "user":
        return playground.run_user_input()
    
    
    if args.subcommand == "eval":
        agent = Agent.from_config("config.json", load_model=args.model_name, eval=True)
        return playground.run_eval_model(args.model_name, agent, args.map_size)
    
    if args.subcommand == "train":
        agent = Agent.from_config("config.json", load_model=args.model_name)
        
        # for posting training progress updates to a discord webhook
        # includes best run gifs, graphs and statistics
        if args.no_webhook:
            webhook = None
        else:
            if args.session_name is None:
                raise Exception("A session name must be specified with --session_name or -s.")
            webhook = Webhook(
                id=os.environ["WEBHOOK_ID"],
                token=os.environ["WEBHOOK_TOKEN"],
                session_name=args.session_name,
                init_msg=f"""Training started, params: 
                    batch size = {agent.batch_size:,}, 
                    memory size = {agent.replay.size:,}, 
                    learning rate = {agent.learning_rate:,},
                    total training steps = {agent.eps_decay_steps:,}, 
                    update target frequency = {agent.update_target_rate:,}"""
            )
            
        return playground.run_builder_server(
            agent=agent,
            webhook=webhook,
            rand_chance=args.rand,
            num_builders=args.num_bots,
            map_size=args.map_size
        )
    
    if args.subcommand == "online":

        # TODO: online train, online eval command options
        act_fn = Agent.eval_pipeline(args.model_name)
        return asyncio.run(playground.run_websocket_server(act_fn))
    
    if args.subcommand == "convert":
        return Agent.convert_train_model(args.model_name)

    raise Exception(f"Unknown subcommand `{args.subcommand}`")

if __name__ == '__main__':
    print(colorama.ansi.clear_screen(), end="")
    print(colorama.Fore.GREEN, end="")
    art.tprint("Deep Polygon", font="tarty4")
    print(colorama.Fore.RESET)
    
    import argparse
    parser = argparse.ArgumentParser()
    
    subcommand = parser.add_subparsers(dest="subcommand", title="Commands")
    train_parser = subcommand.add_parser("train", help="Training with a fixed number of agents. Models saved in ./models/")
    train_parser.add_argument("--no-webhook", help="Disable all webhook initializing and posting", action="store_true", default=False)
    train_parser.add_argument("--model_name", help="Path to model in in ./models directory", default=None)
    train_parser.add_argument("-s", "--session-name", help="Session name for webhook to post")
    train_parser.add_argument("-n", "--num-bots", help="Number of bots to simulate, default: 1", type=int, default=1)
    train_parser.add_argument("-m", "--map-size", help="Size of a square map, default: 200", type=int, default=200)
    train_parser.add_argument("--rand", help="Random action chance", type=float)

    eval_parser = subcommand.add_parser("eval", help="Runs trained model in eval mode, no training only action predictions.")
    eval_parser.add_argument("model_name", help="Path to model in in ./models directory")
    eval_parser.add_argument("-m", "--map-size", help="Size of a square map, default: 120", type=int, default=120)

    user_parser = subcommand.add_parser("user", help="User input environment test. For checking your reward function or fill algorithm.")

    online_parser = subcommand.add_parser("online", help="Runs a websocket server which expects the state in JSON, responds with an action from the loaded model.")
    online_parser.add_argument("model_name", help="A cpu model, see `convert` command.")

    convert_parser = subcommand.add_parser("convert", 
        help="Converts a trained model with 2 NNs, 1 optimizer on cuda device, to a single NN on cpu device for inference."
             "Requires a cuda enabled device.")
    convert_parser.add_argument("model_name", help="Trained model to convert", type=str)

    args = parser.parse_args()
    print(colorama.Fore.RED + colorama.Style.BRIGHT + f"Loading [{args.subcommand}]")
    print(colorama.Fore.RESET, end="")
    try:
        main(args)  
    except KeyboardInterrupt:
        print()
        print(Style.BRIGHT + Fore.GREEN + "Exited!" + Style.RESET_ALL)
        print()
