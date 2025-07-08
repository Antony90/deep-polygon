import { TrainingProgress } from "@/types/message";
import { configureStore, createSlice, PayloadAction } from "@reduxjs/toolkit";

const initialState: TrainingProgress  = {
  steps: 0,
  percent_done: 0,
  eta: "eta",
  rate: 0,
  runtime: "runtime",
  gpu_util: null,
  cpu_util: null,
  epsilon: 0,
};

const trainingProgressSlice = createSlice({
  name: "trainingProgress",
  initialState,
  reducers: {
    setTrainingProgress(state, action: PayloadAction<TrainingProgress>) {
      return action.payload;
    },
  },
});

export const { setTrainingProgress } = trainingProgressSlice.actions;
export default trainingProgressSlice.reducer;
