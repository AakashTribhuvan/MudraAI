// 100 ms chunks of 16-bit PCM from a 16 kHz AudioContext (the context resamples the tab audio)
class Pcm16 extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(1600); this.n = 0 }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        const v = Math.max(-1, Math.min(1, ch[i]))
        this.buf[this.n++] = v < 0 ? v * 32768 : v * 32767
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.buffer, [this.buf.buffer])
          this.buf = new Int16Array(1600)
          this.n = 0
        }
      }
    }
    return true
  }
}
registerProcessor('mudra-pcm16', Pcm16)
