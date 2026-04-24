import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/app/",
  // Dedupe React so dynamic-imported chunks (auth adapters) share the same
  // React instance as the main app. Without this, `@supabase/auth-ui-react`
  // loaded in the supabase adapter chunk can pick up its own React copy,
  // producing the classic "Cannot read properties of null (reading
  // 'useState')" error when its hooks run against a null dispatcher.
  resolve: {
    dedupe: ["react", "react-dom"],
  },
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
