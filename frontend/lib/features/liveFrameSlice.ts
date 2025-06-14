import { LiveFrame } from "@/types/message";
import { createSlice, PayloadAction } from "@reduxjs/toolkit";

const initialState: LiveFrame = {
  img: "",
  reward: 0,
  total_reward: 0,
  ep_length: 0,
  kills: 0,
  land_captured: 0,
  rank: 0
};


const liveFrameSlice = createSlice({
  name: "liveFrame",
  initialState,
  reducers: {
    setLiveFrame(state, action: PayloadAction<LiveFrame>) {
      return action.payload;
    },
  },
});

export const { setLiveFrame } = liveFrameSlice.actions;
export default liveFrameSlice.reducer;
