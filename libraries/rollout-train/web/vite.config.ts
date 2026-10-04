/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The page is built into the Python package, which serves it: `rollout monitor WHERE` needs no Node. `npm run dev`
// serves it here with hot reloading, and passes what it asks for to a monitor on port 8765 (MONITOR names another).
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../src/rollout_train/monitor/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
  server: {
    proxy: { "/api": process.env.MONITOR ?? "http://127.0.0.1:8765" },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.tsx", "src/**/*.test.ts"],
  },
});
