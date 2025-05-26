from dataclasses import asdict, dataclass
import time
from typing import ClassVar, Optional, TypedDict, Union

from agent import Agent
from tqdm import tqdm
import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
except pynvml.NVMLError:
    GPU_AVAILABLE = False
    print("Cannot get GPU utilization for non-NVIDIA gpu")

"""
Message Type Semantics:

- Messages ending with *_update:
    Incrementally modify or merge with existing frontend state (e.g., updating fields of a current object).

- All other message types:
    Replace the corresponding frontend state entirely with the received payload.

This helps distinguish partial vs full state updates across the WebSocket communication layer.
"""


class WsMessage(TypedDict):
    """Sent directly to WebSocket clients"""

    type: str
    payload: dict[str, any]


@dataclass
class Payload:
    type: ClassVar[str]

    def to_message(self) -> WsMessage:
        return WsMessage(type=self.type, payload=asdict(self))



@dataclass
class MeanStatistics(Payload):
    type: ClassVar[str] = "mean_statistics"

    @dataclass
    class MeanStat:
        mean: float
        values: list
        
    @dataclass
    class AvgEpisodeReward(MeanStat):
        num_kills: int
        land_captured: int

    @dataclass
    class AvgEpisodeLength(MeanStat):
        max: int
        min: int

    @dataclass
    class AvgLoss(MeanStat):
        max: int
        std_dev: float

    avg_ep_reward: AvgEpisodeReward
    avg_ep_length: AvgEpisodeLength
    avg_loss: AvgLoss

@dataclass
class GraphUpdate(Payload):
    type: ClassVar[str] = "graph_update"
    graph_name: str  # TODO: update all graphs at once, similar to Statistics attrs
    value: int | float  # to append to frontend


@dataclass
class LiveFrameData(Payload):
    type: ClassVar[str] = "live_frame"
    reward: float
    total_reward: float
    ep_length: int
    kills: int
    land_captured: int
    rank: int
    img: Optional[str] = None

@dataclass
class LeaderboardUpdate(Payload):
    """
    Posts an updated version of the leaderboard, truncated to e.g. top 20 entries
    """
    type: ClassVar[str] = "leaderboard"
    
    agent_name: str
    total_reward: float
    kills: int
    land: int
    rank: int
    timestamp: int


@dataclass
class TrainingProgress(Payload):
    """Updated progress bar attributes, and resource usage"""

    type: ClassVar[str] = "training_progress"
    # training seed and total steps left out since static
    steps: int  # steps done
    percent_done: int  # percent done
    eta: str
    rate: int  # iterations/sec
    runtime: str
    gpu_util: int  # percent
    cpu_util: int  # percent
    epsilon: float  # exploration rate


    @staticmethod
    def _get_gpu_utilization():
        if not GPU_AVAILABLE:
            return "n/a"
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # first GPU
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu  # percentage int
        except Exception:
            return "n/a"


    @staticmethod
    def _get_cpu_utilization():
        # NOTE: Blocks thread for 1 second to measure util accurately
        # Returns CPU percent usage averaged over 1 second
        return int(psutil.cpu_percent(interval=1))


    @classmethod
    def generate_payload(cls, progress: tqdm, agent: "Agent"):
        """
        Every 5s send the training progress update
        """
        # Extract info from pbar
        steps_done = int(progress.n)  # current iteration
        percent_done = int(progress.n / progress.total * 100) if progress.total else 0

        rate = int(progress.format_dict.get("rate", 0))  # iterations/sec from tqdm
        if rate and progress.total:
            remaining_seconds = (progress.total - progress.n) / rate
        else:
            remaining_seconds = 0

        eta = time.strftime(
            "%H:%M:%S", time.gmtime(remaining_seconds)
        )  # estimated time until complete

        elapsed_seconds = progress.format_dict.get("elapsed", 0)
        runtime = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))  # elapsed time

        # For GPU/CPU util, either get from agent or other monitoring lib (stub here)
        gpu_util = cls._get_gpu_utilization()
        cpu_util = cls._get_cpu_utilization()  # blocking

        # Epsilon from agent
        epsilon = agent.get_eps()

        return TrainingProgress(
            steps=steps_done,
            percent_done=percent_done,
            eta=eta,
            rate=rate,
            runtime=runtime,
            gpu_util=gpu_util,
            cpu_util=cpu_util,
            epsilon=epsilon,
        )




# class PlayerState(TypedDict):
#     player_id: str
#     total_reward: float
#     land: str
#     kills: int

# class PlayersUpdate(TypedDict):
#     players: list[PlayerState]
