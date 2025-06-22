import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import WebSocketProvider from "@/components/providers/WebSocketProvider";
import StoreProvider from "@/components/providers/StoreProvider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Deep Polygon",
  description:
    "Real-time dashboard for deep reinforcement learning, featuring live interaction with agents and monitoring",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {

  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <StoreProvider initialState={{}}>
          <WebSocketProvider>{children}</WebSocketProvider>
        </StoreProvider>
      </body>
    </html>
  );
}
