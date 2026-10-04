import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";
import { newQueryClient } from "./api/queries";
import { App } from "./App";
import "./styles.css";

// Places are after the `#` (`#/run/NAME/group/3`): links to the page keep working wherever the monitor is served from,
// behind a proxy's path or not, and the monitor serves one page.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={newQueryClient()}>
      <HashRouter>
        <App />
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
