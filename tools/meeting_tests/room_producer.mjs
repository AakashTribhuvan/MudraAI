// Streams a 16 kHz mono 16-bit WAV to /asr?room=R in real time (100 ms chunks), like the extension's offscreen page.
import { readFileSync } from 'node:fs'
const [wav, room, delayMs = '0'] = process.argv.slice(2)
const pcm = readFileSync(wav).subarray(44)
await new Promise((r) => setTimeout(r, Number(delayMs)))
const ws = new WebSocket(`ws://127.0.0.1:8010/asr?room=${room}`)
ws.binaryType = 'arraybuffer'
await new Promise((r) => ws.addEventListener('open', r))
const t0 = Date.now()
console.log('PRODUCER_START', t0)
for (let i = 0; i < pcm.length; i += 3200) {
  ws.send(pcm.subarray(i, i + 3200))
  const due = t0 + ((i / 3200) + 1) * 100
  await new Promise((r) => setTimeout(r, Math.max(0, due - Date.now())))
}
ws.send(JSON.stringify({ type: 'stop' }))
await new Promise((r) => setTimeout(r, 1500))
ws.close()
console.log('PRODUCER_END', Date.now())
