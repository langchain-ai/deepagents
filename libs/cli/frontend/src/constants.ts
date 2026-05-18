import { getRuntimeConfig } from "./runtimeConfig";

const cfg = getRuntimeConfig();

export const APP_NAME = cfg.appName;
export const APP_SUBTITLE = cfg.subtitle ?? "Your deep agent, deployed.";
export const ASSISTANT_ID = cfg.assistantId;
export const PROMPTS: readonly string[] =
  cfg.prompts && cfg.prompts.length > 0
    ? cfg.prompts
    : [
        "What can you help me research?",
        "Research a topic for me",
        "Help me write a report",
      ];
