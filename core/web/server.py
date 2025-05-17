from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from web.websocket import WebsocketHandler


class WebServer:
    """WebSocket and JSON http API for real-time data and controlling the training or
    simulation parameters during runtime.
    """

    @staticmethod
    def create_app(ws_handler: WebsocketHandler):
        app = FastAPI()
        
        @app.get("/")
        def get_root():
            return { "this": "is a test." }
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            websocket.accept()
            try:
                await ws_handler.serve(websocket)
            except WebSocketDisconnect:
                print("WebSocket client disconnected.")
                
                
        return app
                