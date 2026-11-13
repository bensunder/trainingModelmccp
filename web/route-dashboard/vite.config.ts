import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Served by the gateway at /dashboard (same origin as /admin/v1).
export default defineConfig({
  base: "/dashboard/",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false, target: "es2022", chunkSizeWarningLimit: 300 },
  server: { proxy: { "/admin": "http://127.0.0.1:8090", "/readyz": "http://127.0.0.1:8090" } },
  test: { environment: "jsdom", globals: true, restoreMocks: true },
});
