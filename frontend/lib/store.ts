import { combineReducers, configureStore } from "@reduxjs/toolkit";

import trainingProgress from "@/lib/features/trainingProgressSlice";
import liveFrame from "@/lib/features/liveFrameSlice";

const rootReducer = combineReducers({
  trainingProgress,
  liveFrame,
});

// Infer state from the reducers
export type RootState = ReturnType<typeof rootReducer>;

// Accept an initial state from server-side props
export const makeStore = (preloadedState: Partial<RootState>) => {
  return configureStore({
    reducer: rootReducer,
    preloadedState,
  });
};

// Infer the type of makeStore
export type AppStore = ReturnType<typeof makeStore>;

// Infer the  `AppDispatch` type from the store itself
export type AppDispatch = AppStore["dispatch"];
