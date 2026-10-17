# macOS installation

Native installation is recommended on Apple Silicon so Ollama can use the Mac GPU/unified-memory path directly.

## 1. Ollama

Install Ollama from the official macOS package and pull the default model:

```bash
ollama pull gpt-oss:20b
curl http://127.0.0.1:11434/api/version
```

## 2. Engine

```bash
git clone https://github.com/AlitaYoko99/alitayoko-conversation-engine.git
cd alitayoko-conversation-engine

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

Set a new `AY_API_KEY` in `.env`.

Start:

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

## 3. Optional LaunchAgent

For a permanent service, use a stable checkout path such as:

```text
~/alitayoko-conversation-engine
```

Create `~/Library/LaunchAgents/com.alitayoko.conversation-engine.plist` and use absolute paths for both the Python executable and working directory.

Example structure (replace `YOUR_HOME`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.alitayoko.conversation-engine</string>
  <key>ProgramArguments</key>
  <array>
    <string>YOUR_HOME/alitayoko-conversation-engine/.venv/bin/python</string>
    <string>-m</string>
    <string>uvicorn</string>
    <string>app.main:app</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8787</string>
  </array>
  <key>WorkingDirectory</key>
  <string>YOUR_HOME/alitayoko-conversation-engine</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
```

Load it in the logged-in user session:

```bash
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.alitayoko.conversation-engine.plist
launchctl kickstart -k "gui/$(id -u)/com.alitayoko.conversation-engine"
```

## 4. Public HTTPS

Keep the engine local and use a reverse proxy. See [reverse-proxy.md](reverse-proxy.md).
