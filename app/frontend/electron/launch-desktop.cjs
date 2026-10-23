const { spawn } = require('node:child_process');

// Some managed development environments set this globally, which makes the
// Electron executable behave like plain Node and prevents the desktop window.
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(require('electron'), ['.'], {
  cwd: process.cwd(),
  env,
  stdio: 'inherit',
});

child.on('error', (error) => {
  console.error('Could not start the Electron desktop runtime:', error.message);
  process.exitCode = 1;
});
child.on('exit', (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});
