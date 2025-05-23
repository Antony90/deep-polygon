

from asyncio import Queue as AsyncQueue
import base64
from collections import deque
from io import BytesIO
from queue import Queue as BlockingQueue
import threading

from fastapi import WebSocket
import numpy as np

from constants import STATE_SHAPE
from env.state import GridState

CLIENT_FPS = 10
BUFFER_TIME = 120 # seconds of frames
BUFFER_SIZE = BUFFER_TIME * CLIENT_FPS

class ClientFrameBuffer(AsyncQueue):
    def __init__(self, maxlen: int):
        super().__init__(maxlen)
        self.fill = False

class RenderManager:
    """
    Tracks the highest reward alive player
    Stores a history of the new best replays
    """

    def __init__(self):
        self.player_id = 1
        self.state_queue: AsyncQueue[tuple[str, float]] = AsyncQueue()
        self.render_queue: BlockingQueue[tuple[np.ndarray, float]] = BlockingQueue()
        
        # track the latest state since the render worker thread
        # is (under normal circumstances) faster than the train loop
        self.latest_state: tuple | None = None
        
        # If the render worker falls behind, it will loose the latest frame
        # instead of queuing them up or holding back the train thread
        # This prioritises the train thread, and never pauses it
        
        # Simple event for waiting until latest_state is set
        self.latest_state_ev = threading.Event()
        print(self.latest_state_ev)
        
        self.run = True
        self.clients: dict[WebSocket, ClientFrameBuffer] = {}
        

        
    def queue_state(self, state: any, reward: float):
        grid_state, _ = state
        self.latest_state = (grid_state, reward)

        if not self.latest_state_ev.is_set():
            self.latest_state_ev.set()

        
    def process_render_queue(self):
        """Continuously consume the state queue, converting the
        array into a base64 encoded image string, submitting this to
        the state queue, which is consumed by a client."""
        
        while self.run:
            self.latest_state_ev.wait()
            
            # "pop" the state
            state = self.latest_state
            self.latest_state = None
            
            grid_state, reward = state
            
            self.latest_state_ev.clear()

            img = GridState.to_img(grid_state, dark_mode=False, size=STATE_SHAPE[:2])
            
            # TODO: optimize packet size
            buffer = BytesIO()
            img.save(buffer, format="webp", lossless=False, exact=False, quality=5)
            img_bytes = buffer.getvalue()
            
            img.close()
            buffer.close()
            
            img_b64_bytes = base64.b64encode(img_bytes)
            img_b64_str = "data:image/webp;base64," + img_b64_bytes.decode()
            
            state = (img_b64_str, reward)
            
            # Try to update all client buffers with latest state
            for buf in self.clients.values():
                if buf.full():
                    # Stop filling, let the client drain buffer
                    buf.fill = False
                    
                elif buf.empty():
                    # Enable filling again, client has drained buffer
                    buf.fill = True
                    
                # Cannot be full here since this is the only producer
                # For this buffer
                if buf.fill:
                    buf.put_nowait(state)
    
    async def get_next_state(self, ws_client: WebSocket):
        """Pull from a per-client buffer, which refreshes to the latest
        frames once emptied from full size (`BUFFER_SIZE`)."""
        buf = self.clients[ws_client]
        
        return await buf.get()
    
    def empty(self, ws_client: WebSocket):
        return self.clients[ws_client].empty()
    
    def register_client(self, ws_client: WebSocket):
        self.clients[ws_client] = ClientFrameBuffer(BUFFER_SIZE)
        
    def unregister_client(self, ws_client: WebSocket):
        del self.clients[ws_client]
        
    
    def debug(self):
        return f"render: {self.render_queue_size()}, state: {self.state_queue_size()}"
    
    def render_queue_size(self):
        return self.render_queue.qsize() # approximate
    
    def state_queue_size(self):
        return self.state_queue.qsize() # exact

    
    def get_best_score_replay(self) -> list[GridState]:
        return

    def set_spectate_player(self, player_id):
        # TODO: validation on `env` to check if player exists
        self.player_id = player_id