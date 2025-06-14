import { TrainingProgress } from "@/types/message";
import { configureStore, createSlice, PayloadAction } from "@reduxjs/toolkit";

const initialState: TrainingProgress | null = {
  steps: 1,
  percent_done: 1,
  eta: "eta",
  rate: 1,
  runtime: "runtime",
  gpu_util: 1,
  cpu_util: 1,
  epsilon: 1,
};

export const mockServerSideState: TrainingProgress = {
  steps: 0,
  percent_done: 0,
  eta: "eta_ssr",
  rate: 0,
  runtime: "runtime_ssr",
  gpu_util: 0,
  cpu_util: 0,
  epsilon: 0,
};

const trainingProgressSlice = createSlice({
  name: "trainingProgress",
  initialState: null as TrainingProgress | null,
  reducers: {
    setTrainingProgress(state, action: PayloadAction<TrainingProgress | null>) {
      return action.payload;
    },
  },
});

export const { setTrainingProgress } = trainingProgressSlice.actions;
export default trainingProgressSlice.reducer;
