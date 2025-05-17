import asyncio

from train.manager import RenderManager
from fastapi import WebSocket

class WebsocketHandler:
    def __init__(self, render_manager: RenderManager):
        self.render_manager = render_manager

    async def serve(self, websocket: WebSocket):
        listen_task = asyncio.create_task(self.listen(websocket))
        send_task = asyncio.create_task(self.send_state_forever(websocket))
        
        # Stop server if any handler exits
        _, pending = await asyncio.wait(
            [listen_task, send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        

    async def send_state_forever(self, websocket: WebSocket, delay=0.1):
        """Continuously read from the state queue and send to client.
        
        `delay` (seconds) between each message.
        """
        while True:
            if not self.render_manager.empty():
                b64_img_bytes, reward = self.render_manager.pop_state()
                # Send the image as base64-encoded PNG to the client
                await websocket.send({"image": b64_img_bytes, "reward": reward})
            await asyncio.sleep(delay)
        
            
    async def listen(self, websocket: WebSocket):
        # TODO: just use binary
        async for message in websocket.iter_text():
            player_id = int(message)
            self.render_manager.set_spectate_player(player_id)
