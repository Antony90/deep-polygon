# Frontend


## Feature Overview 
Live dashboard for the training server.

- Real-time Monitoring via the [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
    - Live Spectator View
        - Streams base64 encoded frames at low resolution
        - Supports switching between
            - Current alive best agent
            - Any of the agents and their players
            - Historical replay of episodes, such as the best overall, or best over a recent period
    - Live Training Metrics
        - Graphed Stats updated in real-time
            - Advantage value of the optimal action (Q value)
            - Episode total reward
            - Mean loss
            - Episode length
            - Number of training steps completed
- Server Controls
    - Model checkpoints
        - Save model 
        - List `.pt` files
        - Download any model
    - Pause/resume training
    - Training session restart with progress saved
    - Download `.gif` replays
    - View Agent configuration and hyperparameters

- REST API
    - Pause/resume/save training state
    - Get model files
    - Download replay files as GIF
    - Overview of training (hyper) parameters

## Tech Overview
- Next.js (App  Router)
- TailwindCSS v4
- [shadcn](https://ui.shadcn.com/) Component framework
- WebSocket client
- Modern Graphing UI
             