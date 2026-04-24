/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_NEW_CHAT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
