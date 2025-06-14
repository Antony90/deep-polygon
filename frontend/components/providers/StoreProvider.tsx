"use client";
import { useRef } from "react";
import { Provider } from "react-redux";
import { makeStore, AppStore, RootState } from "@/lib/store";

export default function StoreProvider({
  initialState,
  children,
}: {
  initialState: Partial<RootState>;
  children: React.ReactNode;
}) {
  const storeRef = useRef<AppStore | null>(null);
  if (!storeRef.current) {
    storeRef.current = makeStore(initialState);
  }

  return <Provider store={storeRef.current}>{children}</Provider>;
}
