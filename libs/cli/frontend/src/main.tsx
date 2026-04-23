import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { APP_NAME } from "./constants";
import "./index.css";

document.title = APP_NAME;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
