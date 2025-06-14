import asyncio
from dataclasses import asdict

from train.render import CLIENT_FPS, RenderManager
from fastapi import WebSocket

from web.message import LiveFrameData, Payload


class WebSocketHandler:
    def __init__(self, render_manager: RenderManager):
        self.render_manager = render_manager
        self.clients: set[WebSocket] = set()

        self.broadcast_payload_queue: asyncio.Queue[Payload] = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

        # Cache the latest payload of each type, to expose this to
        # The REST API as a GET request endpoint
        self._latest_payload_cache: dict[str, Payload] = dict()

    async def serve(self, websocket: WebSocket):
        # Continuous loop
        listen_task = asyncio.create_task(self.listen(websocket))
        send_state_task = asyncio.create_task(self.send_state_forever(websocket))

        # Stop server if any handler exits
        done, pending = await asyncio.wait(
            [listen_task, send_state_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    def register(self, websocket: WebSocket):
        self.clients.add(websocket)

    def unregister(self, websocket: WebSocket):
        self.clients.discard(websocket)

    async def send_state_forever(self, ws_client: WebSocket, delay=0.1):
        """
        Continuously read from the state queue and send to client.

        `delay` (seconds) between each message.
        """
        while True:  # TODO: self.run var
            if not self.render_manager.empty(ws_client):
                live_frame = await self.render_manager.get_next_state(ws_client)
                msg = live_frame.to_message()

                await ws_client.send_json(msg)

            await asyncio.sleep(1 / CLIENT_FPS)

    async def listen(self, ws_client: WebSocket):
        # TODO: just use binary
        async for message in ws_client.iter_text():
            player_id = int(message)
            self.render_manager.set_spectate_player(player_id)

    def put_broadcast_payload(self, payload: Payload):
        """
        Can be called from a synchronous context. Will error if has a max size and is full
        """
        self.broadcast_payload_queue.put_nowait(payload)
        self._set_latest_payload(payload)

    async def run_broadcast_loop(self):
        """
        Continuously waits for payloads, and broadcasts to all connected clients.
        Example: broadcasting new statistic values, graph updates, leaderboard changes
        """
        while not self._shutdown_event.is_set():
            payload = await self.broadcast_payload_queue.get()
            msg = payload.to_message()

            await asyncio.gather(
                *[ws_client.send_json(msg) for ws_client in self.clients]
            )

    def stop_broadcast_loop(self):
        self._shutdown_event.set()

    def _set_latest_payload(self, payload: Payload):
        """
        Update latest payload cache for the type.
        """
        if payload.type != LiveFrameData.type:
            self._latest_payload_cache[payload.type] = payload

    def get_latest_payloads(self, exclude_types=[LiveFrameData.type]):
        return {
            type: asdict(payload)
            for type, payload in self._latest_payload_cache.items()
            if type not in exclude_types
        }
        
    def get_latest_payload(self, payload_type):
        payload = self._latest_payload_cache[payload_type]
        
        return asdict(payload)
