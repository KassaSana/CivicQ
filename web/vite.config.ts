import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Relative base so the build works from any static host path (e.g. GitHub Pages)
export default defineConfig({
  base: './',
  plugins: [react()],
  test: { environment: 'node', testTimeout: 60000 },
});
