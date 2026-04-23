import { getRuntimeConfig } from "./runtimeConfig";

const cfg = getRuntimeConfig();

export const APP_NAME = cfg.appName;
export const APP_DESCRIPTION = "Your deep agent, deployed.";
export const ASSISTANT_ID = cfg.assistantId;
