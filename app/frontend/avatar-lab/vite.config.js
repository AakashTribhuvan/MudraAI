import path from 'node:path'
import { createReadStream, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const here = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(here, '..')
const modelPath = path.join(frontendRoot, 'public', 'ybot.glb')

function avatarModelAsset() {
  return {
    name: 'avatar-lab-model-asset',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url?.split('?')[0] !== '/ybot.glb') return next()
        response.setHeader('Content-Type', 'model/gltf-binary')
        createReadStream(modelPath).pipe(response)
      })
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'ybot.glb', source: readFileSync(modelPath) })
    },
  }
}

export default defineConfig({
  root: here,
  plugins: [react(), avatarModelAsset()],
  publicDir: false,
  server: {
    port: 5173,
    strictPort: true,
    open: false,
    fs: { allow: [frontendRoot] },
  },
  build: { outDir: path.join(here, 'dist'), emptyOutDir: true },
})
