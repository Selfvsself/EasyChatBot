# EasyChatBot
A modular chatbot with a user-friendly web front-end that lets you instantly pick a chatting mode: a simple conversational bot or an AI partner for practicing English speaking.

## Run On Ubuntu Server With Docker

This project is configured to run as a single container (`API + internal worker` in one process).
External services are expected to run on other machines:
- PostgreSQL
- Kafka
- Ollama

### 1. Install Docker and Docker Compose plugin

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

### 2. Clone project and configure environment

```bash
git clone <your-repo-url>
cd EasyChatBot
cp .env .env.backup 2>/dev/null || true
```

Edit `.env` with your external service addresses:

```env
APP_HOST=0.0.0.0
APP_PORT=8000

DATABASE_URL=postgresql://<user>:<password>@<postgres-host>:5432/<db>
KAFKA_BOOTSTRAP_SERVERS=<kafka-host>:9092
LLM_URL=http://<ollama-host>:11434
```

Important:
- Use reachable IP/DNS names from the Ubuntu server.
- Open firewall routes from Ubuntu to PostgreSQL/Kafka/Ollama.
- Do not commit real tokens/passwords to git.

### 3. Build and run

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f easychatbot
```

App should be available on:
- `http://<ubuntu-server-ip>:8000`

## Autostart On System Boot

There are two good options.

### Option A (recommended): systemd unit for compose stack

Create `/etc/systemd/system/easychatbot.service`:

```ini
[Unit]
Description=EasyChatBot Docker Compose Stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/EasyChatBot
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
RemainAfterExit=yes
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now easychatbot.service
sudo systemctl status easychatbot.service
```

Notes:
- Set `WorkingDirectory` to your real project path.
- This gives explicit service control via `systemctl` and predictable boot behavior.

### Option B: rely on Docker restart policy only

`docker-compose.yml` already uses:
- `restart: unless-stopped`

If Docker daemon starts on boot (`systemctl enable docker`), the container is usually restored automatically.
This is simpler, but less explicit than a dedicated systemd unit for the app.

## Useful Operations

```bash
# rebuild after code changes
docker compose up -d --build

# restart app only
docker compose restart easychatbot

# view logs
docker compose logs -f easychatbot

# stop/start
docker compose down
docker compose up -d
```
