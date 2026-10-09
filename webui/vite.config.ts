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
      // changeOrigin: false — Host проксируется как есть (origin браузера,
      // напр. :5173). Иначе Host меняется на :8790 и CSRF-guard (#87) отклоняет
      // POST, сравнивая Origin (:5173) с Host (:8790).
      '/api': { target: 'http://127.0.0.1:8790', changeOrigin: false },
    },
  },
  build: {
    outDir: fileURLToPath(new URL('../src/audio_transcriber/web/static', import.meta.url)),
    emptyOutDir: true,
  },
})
