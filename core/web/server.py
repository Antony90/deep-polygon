from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tqdm import tqdm

from train.render import RenderManager
from web.websocket import WebsocketHandler


class WebServer:
    """WebSocket and JSON http API for real-time data and controlling the training or
    simulation parameters during runtime.
    """

    @staticmethod
    def create_app(ws_handler: WebsocketHandler, render_manager: RenderManager, pbar: tqdm):
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"]
        )
        
        @app.get("/queue")
        def queue_size():
            return {
                "work_size": render_manager.render_queue_size(),
                "state_size": render_manager.state_queue_size()
            }
            
        @app.get("/progress")
        def progress():
            return { "rate": int(pbar.format_dict["rate"]) }
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            
            render_manager.register_client(websocket)
            try:
                await ws_handler.serve(websocket)
            except WebSocketDisconnect:
                print("WebSocket client disconnected.")
            render_manager.unregister_client(websocket)
                
                
        return app
                