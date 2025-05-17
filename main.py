import asyncio
import json
import sys
import os
import time
import art
import numpy as np

import colorama
from colorama import Fore, Style
import torch
from tqdm import tqdm
from threading import Event, Thread
from queue import Queue, Empty
from typing import Callable, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import uvicorn

from env.player.builder import Builder
from train.manager import AgentPlayerGroup, RenderManager, TrainingManager
from train.stats import TrainingStats
from web.server import WebServer
from web.webhook import Webhook
from web.websocket import WebsocketHandler
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


def setup_training(args):
    agent = Agent.from_config("config.json", load_model=args.model_name)
    pbar = tqdm(total=agent.eps_decay_steps, leave=False)
    stats = TrainingStats()
    
    # for posting training progress updates to a discord webhook
    # includes best run gifs, graphs and statistics
    if args.no_webhook:
        webhook = None
    else:
        if args.session_name is None:
            raise Exception("A session name must be specified with --session_name or -s.")
        if not os.environ.get("WEBHOOK_ID") or not os.environ.get("WEBHOOK_TOKEN"):
            raise Exception("Webhook ID or Token not specified. To run the project without webhook logging, use the --no-webhook option.")
        
        webhook = Webhook(
            id=os.environ["WEBHOOK_ID"],
            token=os.environ["WEBHOOK_TOKEN"],
            session_name=args.session_name,
            stats=stats,
            progress_bar=pbar
        )
    
    render_manager = RenderManager()
    
    def run_webserver():
        ws_handler = WebsocketHandler(render_manager)
        app = WebServer.create_app(ws_handler)
        app.host()
        uvicorn.run(app, host="0.0.0.0", port=8000)

    def run_training():
        builder_group = AgentPlayerGroup(agent, Builder, args.num_bots)
        training_manager = TrainingManager(args.map_size, [builder_group], stats, pbar, render_manager, webhook)
        training_manager.start(pbar)
        
    webserver_thread = Thread(target=run_webserver, daemon=True)
    webserver_thread.start()
    
    training_thread = Thread(target=run_training, daemon=True)
    training_thread.start()
    
    
    training_thread.join()
    print("Training exited")
    
    webserver_thread.join()
    print("Webserver exited.")
    

def main(args):
    playground = Playground()
    if args.subcommand == "user":
        return playground.run_user_input()
    
    
    if args.subcommand == "eval":
        agent = Agent.from_config("config.json", load_model=args.model_name, eval=True)
        return playground.run_eval_model(args.model_name, agent, args.map_size)
    
    if args.subcommand == "train":
        setup_training(args)
    
    if args.subcommand == "online":

        # TODO: online train, online eval command options
        act_fn = Agent.eval_pipeline(args.model_name)
        return asyncio.run(playground.run_websocket_server(act_fn))
    
    if args.subcommand == "convert":
        return Agent.convert_train_model(args.model_name)

    raise Exception(f"Unknown subcommand `{args.subcommand}`")

if __name__ == '__main__':
    # print(colorama.ansi.clear_screen(), end="")
    # print(colorama.Cursor.POS(1, 1), end="")  # Move to top left of terminal
    print(colorama.Fore.GREEN, end="")
    art.tprint("Deep Polygon", font="tarty4")
    print(colorama.Fore.RESET)
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.description = "A multi-agent deep reinforcement learning experiment - designing collaborative and competitive agents for grid based arena environments like splix.io, paper.io and tileman.io."
    
    subcommand = parser.add_subparsers(dest="subcommand", title="Commands")
    train_parser = subcommand.add_parser("train", help="Training with a fixed number of agents. Models saved in ./models/")
    train_parser.add_argument("--no-webhook", help="Disable all webhook initializing and posting", action="store_true", default=False)
    train_parser.add_argument("--model-name", "-M", help="Path to a trained model in in ./models directory. Starts training with this model.", default=None)
    train_parser.add_argument("--session-name", "-s", help="Session name for webhook to post", required=True)
    train_parser.add_argument("--num-bots", "-n", help="Number of bots to simulate", type=int, default=1)
    train_parser.add_argument("--map-size", "-m", help="Size of a square map", type=int, default=200)
    train_parser.add_argument("--rand", help="Initial random action chance/epsilon value. Use to resume training or when starting with a competent trained model.", type=float)

    eval_parser = subcommand.add_parser("eval", help="Runs trained model in eval mode, no training only action predictions.")
    eval_parser.add_argument("--model_name", help="Path to model in in ./models directory")
    eval_parser.add_argument("--map-size", "-m", help="Size of a square map, default: 120", type=int, default=120)

    user_parser = subcommand.add_parser("user", help="User input environment test. For checking your reward function or fill algorithm.")

    online_parser = subcommand.add_parser("online", help="Runs a websocket server which expects the state in JSON, responds with an action from the loaded model.")
    online_parser.add_argument("--model_name", help="A cpu model, see `convert` command.")

    convert_parser = subcommand.add_parser("convert", 
        help="Converts a trained model with 2 NNs, 1 optimizer on cuda device, to a single NN on cpu device for inference."
             "Requires a cuda enabled device.")
    convert_parser.add_argument("--model_name", help="Trained model to convert", type=str)

    args = parser.parse_args()

    try:
        main(args)  
    except KeyboardInterrupt:
        print()
        print(Style.BRIGHT + Fore.GREEN + "Exited!" + Style.RESET_ALL)
        print()
