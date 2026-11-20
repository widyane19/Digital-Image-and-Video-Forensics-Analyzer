import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Forward API calls to the FastAPI server (uvicorn app:app --reload).
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
