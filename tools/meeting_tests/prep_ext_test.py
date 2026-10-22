"""Prepares the extension tests (ext_e2e.mjs, ext_real.mjs) in experiments/tools/meeting_tests/ (git-ignored):

- ext_test/: a copy of app/meeting-extension that also runs on the mock meeting (http://127.0.0.1:5599), with a fixed
  extension ID (a test-only key generated here, never committed) and two test hooks in the service worker
  (__mudraTestStart: start capture like the toolbar; __mudraTestShowPanel: show the panel for a caption room).
- mock_meeting/: the mock meeting page plus speech.wav (a Windows voice; made by make_speech_wav.ps1 if missing).

    python tools/meeting_tests/prep_ext_test.py      # prints the test extension ID
    python -m http.server 5599 --bind 127.0.0.1 -d experiments/tools/meeting_tests/mock_meeting
"""

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "experiments" / "tools" / "meeting_tests"
EXT = WORK / "ext_test"
MOCK = WORK / "mock_meeting"
SPEECH = "Hello, how are you? Thank you for joining. I need water, please help me. Good morning my friend, see you tomorrow."

WORK.mkdir(parents=True, exist_ok=True)
shutil.rmtree(EXT, ignore_errors=True)
shutil.copytree(ROOT / "app" / "meeting-extension", EXT)

key_file = WORK / "ext_test_key.pem"
if key_file.exists():
    key = serialization.load_pem_private_key(key_file.read_bytes(), None)
else:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
der = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
ext_id = "".join(chr(ord("a") + int(c, 16)) for c in hashlib.sha256(der).hexdigest()[:32])

m = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
m["key"] = base64.b64encode(der).decode()
for k in ("host_permissions",):
    m[k].append("http://127.0.0.1:5599/*")
m["content_scripts"][0]["matches"].append("http://127.0.0.1:5599/*")
m["web_accessible_resources"][0]["matches"].append("http://127.0.0.1:5599/*")
(EXT / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
cfg = (EXT / "config.js").read_text(encoding="utf-8")
(EXT / "config.js").write_text(cfg.replace("export const SUPPORTED = [", "export const SUPPORTED = [/^http:\\/\\/127\\.0\\.0\\.1:5599\\//, "), encoding="utf-8")
bg = (EXT / "background.js").read_text(encoding="utf-8")
bg += """
// TEST ONLY (tools/meeting_tests/prep_ext_test.py)
globalThis.__mudraTestStart = async (tabId) => { await loaded; await start(await chrome.tabs.get(tabId)); return session }
globalThis.__mudraTestShowPanel = async (tabId, room) => {
  const config = await loadConfig()
  session = { tabId, room }
  await save()
  await setBadge(tabId, true)
  await chrome.tabs.sendMessage(tabId, { type: 'show-panel', room, web: config.web })
  return session
}
"""
(EXT / "background.js").write_text(bg, encoding="utf-8")

MOCK.mkdir(parents=True, exist_ok=True)
shutil.copy(HERE / "mock_meeting" / "index.html", MOCK / "index.html")
if not (MOCK / "speech.wav").exists():
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "make_speech_wav.ps1"),
                    "-Text", SPEECH, "-Wav", str(MOCK / "speech.wav"), "-Json", str(MOCK / "speech_words.json")], check=True)
print(ext_id)
