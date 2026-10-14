import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            if ((err as any).code === 'ECONNREFUSED') {
              if (res && 'writeHead' in res && !(res as any).headersSent) {
                (res as any).writeHead(503, {
                  'Content-Type': 'application/json',
                  'Retry-After': '1',
                });
                (res as any).end(
                  JSON.stringify({
                    error: 'Backend API is still starting up. Please wait a moment...',
                    code: 'BACKEND_STARTING',
                  })
                );
              }
            }
          });
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
});
