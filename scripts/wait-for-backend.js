// scripts/wait-for-backend.js
const url = 'http://127.0.0.1:7860/api/options';
const maxAttempts = 30; // 15 seconds max
const delayMs = 500;

process.stdout.write('[fe] Waiting for backend API at http://127.0.0.1:7860 to be ready...');

for (let i = 0; i < maxAttempts; i++) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(1000) });
    if (res.ok || res.status < 500) {
      console.log('\n[fe] Backend API is ready! Launching frontend dev server...');
      process.exit(0);
    }
  } catch (_) {
    process.stdout.write('.');
  }
  await new Promise((resolve) => setTimeout(resolve, delayMs));
}

console.warn('\n[fe] Backend wait timed out, launching frontend anyway...');
process.exit(0);
