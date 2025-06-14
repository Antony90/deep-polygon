import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from tqdm import tqdm
from web.message import PAYLOAD_TYPES

from train.render import RenderManager
from web.websocket import WebSocketHandler


class WebServer:
    """WebSocket and JSON http API for real-time data and controlling the training or
    simulation parameters during runtime.
    """

    @staticmethod
    def create_app(
        ws_handler: WebSocketHandler, render_manager: RenderManager, pbar: tqdm
    ):
        # Run websocket broadcast handler before startup and waits for shutdown
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            asyncio.create_task(ws_handler.run_broadcast_loop())
            yield  # Wait for shutdown
            ws_handler.stop_broadcast_loop()

        app = FastAPI(lifespan=lifespan)
        app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"])

        @app.get("/latest")
        def latest_all():
            return ws_handler.get_latest_payloads()

        @app.get("/latest/{payload_type}")
        def latest(payload_type: str):
            if payload_type not in PAYLOAD_TYPES:
                raise HTTPException(
                    status_code=400, detail=f"Invalid payload type {payload_type}"
                )

            try:
                return ws_handler.get_latest_payload(payload_type)
            except KeyError:
                raise HTTPException(status_code=404, detail="Payload not found")

        @app.get("/progress")
        def progress():
            return {"rate": int(pbar.format_dict["rate"])}

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()

            ws_handler.register(websocket)
            render_manager.register_client(websocket)

            try:
                await ws_handler.serve(websocket)
            except WebSocketDisconnect:
                print("WebSocket client disconnected.")
            ws_handler.unregister(websocket)
            render_manager.unregister_client(websocket)

        return app
