import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'

// Сборка кладётся прямо в пакет Python: собранный SPA раздаёт тот же сервер.
// В dev-режиме /api проксируется на локальный сервер транскрибера.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
  build: {
    outDir: fileURLToPath(new URL('../src/audio_transcriber/web/static', import.meta.url)),
    emptyOutDir: true,
  },
})
