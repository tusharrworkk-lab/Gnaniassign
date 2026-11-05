import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative assets keep the compiled frontend portable for static hosting and
// Flutter's local WebView asset loader.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
