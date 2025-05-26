

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
from web.message import LiveFrameData

CLIENT_FPS = 10
BUFFER_TIME = 120 # seconds of frames
BUFFER_SIZE = BUFFER_TIME * CLIENT_FPS

class ClientFrameBuffer(AsyncQueue[LiveFrameData]):
    def __init__(self, maxlen: int):
        super().__init__(maxlen)
        self.can_fill = True

class RenderManager:
    """
    Tracks the highest reward alive player
    Stores a history of the new best replays
    """

    def __init__(self):
        self.player_id = 1
        
        # track the latest state since the render worker thread
        # is (under normal circumstances) faster than the train loop
        self.latest_state: LiveFrameData | None = None
        
        # If the render worker falls behind, it will loose the latest frame
        # instead of queuing them up or holding back the train thread
        # This prioritises the train thread, and never pauses it
        
        # Simple event for waiting until latest_state is set
        self.latest_state_ev = threading.Event()
        
        self.run = True
        self.clients: dict[WebSocket, ClientFrameBuffer] = {}
        

        
    def queue_state(self, state: tuple[np.ndarray], frame_data: LiveFrameData):
        grid_state, _ = state # ignoring scalar state
        self.latest_state = grid_state
        self.latest_frame_data = frame_data

        if not self.latest_state_ev.is_set():
            self.latest_state_ev.set()

    def _process_state(self, state: np.ndarray) -> str:
        """
        Render np array into a webp image and encode it as a
        base64 string.
        """
        img = GridState.to_img(state, dark_mode=False, size=STATE_SHAPE[:2])
        
        # TODO: optimize packet size
        buffer = BytesIO()
        img.save(buffer, format="webp", lossless=False, exact=False, quality=5)
        img_bytes = buffer.getvalue()
        
        img.close()
        buffer.close()
        
        img_b64_bytes = base64.b64encode(img_bytes)
        img_b64_str = "data:image/webp;base64," + img_b64_bytes.decode()
        
        return img_b64_str
        
    def process_latest_state_loop(self):
        """Continuously consume the state queue, converting the
        array into a base64 encoded image string, submitting this to
        the state queue, which is consumed by a client."""
        
        while self.run:
            self.latest_state_ev.wait()
            
            # Read the latest state
            state = self.latest_state
            frame_data = self.latest_frame_data
            
            self.latest_state_ev.clear()
            
            # If there are no clients that can accept the latest frame (due to drain cycle)
            # We don't need to process this frame at all
            if not any(buf.can_fill for buf in self.clients.values()):
                continue

            img_b64_str = self._process_state(state)
            
            # Update missing frame attrs
            frame_data.img = img_b64_str
            # TODO: state.rank using Leaderboard

            # Try to update all client buffers with latest state
            for buf in self.clients.values():
                if buf.full():
                    # Stop filling, let the client drain buffer
                    buf.can_fill = False
                    
                elif buf.empty():
                    # Enable filling again, client has drained buffer
                    buf.can_fill = True
                    
                # Cannot be full here since this is the only producer
                # For this buffer
                if buf.can_fill:
                    buf.put_nowait(frame_data)
    
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
        
    
    def get_best_score_replay(self) -> list[GridState]:
        return

    def set_spectate_player(self, player_id):
        # TODO: validation on `env` to check if player exists
        self.player_id = player_id