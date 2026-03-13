import { defineConfig } from 'vite'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [topLevelAwait()],
  root: 'frontend',
  build: { outDir: '../frontend/dist', emptyOutDir: true }
})
