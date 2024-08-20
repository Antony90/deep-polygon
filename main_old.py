"""

Old entrypoint

"""


import datetime, os, math, time, psutil, asyncio, json

from tqdm import tqdm

from webhook import Webhook
from agent import Agent
import numpy as np

import torch as T

import websockets as ws
from websockets.exceptions import ConnectionClosedError
from apscheduler.schedulers.background import BackgroundScheduler


# TODO:
# - make webhook post based on num train steps
# - use classes to group variables and functions better
# - fix inconsistent array state in matplotlib/graphs
# - add other lines to reward graph e.g. mean ep reward, max ep reward, ep length
#       may need to take moving average for each

ep_num = 0
log_every = 500 # episodes
client_cntr = 0

best_reward = -math.inf
mean = -math.inf
best_run = None
avail_percent = 100 - psutil.virtual_memory()[2]

ep_rewards = []
means = []
q_vals = []

grid_dims = (31, 31)
n_channels = 3
grid_shape = (n_channels, *grid_dims)

device = T.device("cuda" if T.cuda.is_available() else "cpu")

agent = Agent(
    grid_shape,
    batch_size=32,
    memory_size=300_000,
    min_memory_size=50_000,
    gamma=0.99,
    learning_rate=0.00005,
    eps_decay_steps=20_000_000,
    update_target_rate=10_000,
    device=device
)
# agent.load("base.pt")
train_pbar = tqdm(total=agent.eps_decay_steps)


webhook_url = f"https://discord.com/api/webhooks/{os.environ['WEBHOOK_ID']}/{os.environ['WEBHOOK_TOKEN']}"
webhook = Webhook(
    webhook_url,
    session_name=os.environ.get('THREAD_NAME'),
    init_msg=
    f"""Training started, params: 
        batch size = {agent.batch_size:,}, 
        memory size = {agent.replay.size:,}, 
        learning rate = {agent.learning_rate:,},
        total training steps = {agent.eps_decay_steps:,}, 
        update target frequency = {agent.update_target_rate:,}"""
)

def post_webhook(new_best: bool):
    global best_reward, best_run, ep_num, ep_rewards, mean, q_vals, avail_percent
    if best_run is None or len(ep_rewards) == 0:
        return
    
    if new_best: # Only generate and send a gif
        replay_gif = Webhook.generate_gif(best_run)
        webhook.send_replay_gif(replay_gif, best_reward)
        replay_gif.close()
    else: # Send the report embed with graphs
        # Get time remaining and elapsed from progress bar
        elapsed = train_pbar.format_dict['elapsed']
        rate = train_pbar.format_dict['rate']
        remaining = (train_pbar.total - train_pbar.n) / \
            rate if rate else 0

        # Generate graphs in and store memory
        reward_graph = Webhook.reward_graph(ep_rewards, means, log_every)
        q_val_graph = Webhook.q_val_graph(q_vals)
        webhook.send_report(agent.get_eps(), best_reward, mean, agent.train_cntr,
                            agent.eps_decay_steps, agent.replay.percent_full(), avail_percent,
                            int(elapsed), int(remaining), reward_graph, q_val_graph)
        reward_graph.close()
        q_val_graph.close()


def check_avail_mem():
    global avail_percent
    avail_percent = 100 - psutil.virtual_memory()[2]
    if avail_percent <= 5.0:
        webhook.send_msg(f'{avail_percent:.2f}% available memory!', alert=True)


def parse_state(state):
    state = np.array(state)
    return state[15:46, 15:46].reshape(grid_shape).astype(np.float32)

train_lock = asyncio.Lock()
memory_lock = asyncio.Lock()
start = time.perf_counter()
first_train = True

async def handle_client(ws):
    global best_reward, best_run, ep_num, ep_rewards, mean, log_every, client_cntr, train_pbar, train_lock, memory_lock, first_train
    client_cntr += 1
    client_id = client_cntr
    webhook.send_msg(f"Client #{client_id} connected")
    
    prev_state = None
    done = False
    ep_reward = 0
    episode_history = []

    try:
        async for message in ws:
            msg = json.loads(message)
            new_state = parse_state(msg['state'])

            # store state for training, get policy action
            action, q_val = agent.act(new_state)
            
            episode_history.append(new_state)
            q_vals.append(q_val)
            
            

            # false if first state of an episode
            if prev_state is not None:
                await ws.send(json.dumps({'action': int(action), 'new_ep': prev_state is None}))
                reward = msg['reward']
                done = msg['done']

                async with memory_lock:
                    agent.replay.store(prev_state, action,
                                       reward, new_state, int(done))
                
                if agent.replay.cntr >= agent.min_memory_size:
                    if first_train:
                        first_train = False
                        webhook.send_msg(
                            f"Filled initial memory with {agent.min_memory_size} transitions, time taken: {datetime.timedelta(seconds=int(time.perf_counter()-start))}")
                    async with train_lock:
                        agent.train()
                    train_pbar.update(1)
                
                ep_reward += reward
            prev_state = new_state

            if done:
                # reset vars for next client episode
                ep_num += 1
                done = False
                prev_state = None
                ep_rewards.append(ep_reward)

                # save the model and update best score
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    best_run = episode_history

                    # spams webhook less at the start
                    if agent.replay.cntr >= agent.min_memory_size:
                        post_webhook(True)

                    agent.save(f"ep{ep_num:06}_t{agent.train_cntr:07}_b{ep_reward:5.3f}.pt")

                # calculate mean and send status report with webhook
                if ep_num % log_every == 0:
                    mean = sum(
                        ep_rewards[len(ep_rewards)-log_every:]) / log_every
                    means.append(mean)
                    agent.save(f"ep{ep_num:06}_t{agent.train_cntr:07}_m{mean:5.3f}.pt")

                ep_reward = 0
                episode_history = []
                # update terminal progress bar
                train_pbar.set_postfix({
                    'eps': "{:.2f}".format(agent.get_eps()), 
                    'maxr': "{:.3f}".format(best_reward), 
                    f'mean_{log_every}r': "{:.3f}".format(mean) if not math.isinf(mean) else mean, 
                    'episode': ep_num, 
                    'replay_mem%': "{:.2f}".format(agent.replay.percent_full())
                })
                
    except ConnectionClosedError:
        webhook.send_msg(f"Client #{client_id} closed connection")




async def main():
    server_entrypoint = ws.serve(handle_client, port=4321) if os.environ['USE_TCP_SOCK'] == '1' \
                   else ws.unix_serve(handle_client, path='/sock/controller.sock')
    async with server_entrypoint:
        await asyncio.Future()

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(post_webhook, 'interval', args=(False,), hours=4, max_instances=1)
    scheduler.add_job(check_avail_mem, 'interval', hours=1, max_instances=1)
    scheduler.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        train = False
        train_pbar.close()
        agent.save(f'final_{ep_num}.pt')

        scheduler.remove_all_jobs()
        scheduler.shutdown(wait=False)
        print("Exited")

    webhook.send_msg("Exited")