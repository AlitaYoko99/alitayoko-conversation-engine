# Linux installation

## 1. Ollama

Install Ollama using the current official Linux installer, then pull the model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
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

Set a unique `AY_API_KEY` in `.env`, then test:

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
```

## 3. systemd example

A typical system service can use:

```ini
[Unit]
Description=AlitaYoko Conversation Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/alitayoko-conversation-engine
ExecStart=/opt/alitayoko-conversation-engine/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Adjust the user/path to your installation, save the unit as `/etc/systemd/system/alitayoko-conversation-engine.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now alitayoko-conversation-engine
sudo systemctl status alitayoko-conversation-engine
```

## 4. Public HTTPS

Do not bind Uvicorn directly to a public interface unless you fully understand the consequences. Prefer a reverse proxy with TLS. See [reverse-proxy.md](reverse-proxy.md).
