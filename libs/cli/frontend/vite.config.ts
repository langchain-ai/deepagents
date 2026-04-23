import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: {
    outDir: "dist",
  },
  server: {
    proxy: {
      "/threads": "http://localhost:2024",
      "/runs": "http://localhost:2024",
      "/assistants": "http://localhost:2024",
    },
  },
});
