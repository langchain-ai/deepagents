import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { APP_NAME } from "./constants";
import { ThemeProvider } from "./ThemeProvider";
import "./index.css";

document.title = APP_NAME;

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("Missing #root element in index.html");

createRoot(rootEl).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
