# Deployment Guide (Backend + PostgreSQL)

This guide covers:
- PostgreSQL setup (with pgvector)
- Backend deployment on Windows (recommended for this workspace)
- Linux deployment option (systemd + gunicorn)
- Docker Compose deployment
- Smoke checks after deployment

## Quick Cloud Deploy (Ubuntu VPS, 24/7)

If you want the fastest path to run continuously without your laptop, use this section.

### 0) Create a cloud VM

- Provider examples: DigitalOcean, Hetzner, AWS EC2, Azure VM.
- OS: Ubuntu 22.04 or 24.04.
- Size: at least 2 vCPU / 4 GB RAM for initial production usage.

### 1) SSH and install Docker

```bash
ssh <user>@<server_ip>
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

### 2) Copy code to server

```bash
git clone <your_repo_url> rag-chatbot
cd rag-chatbot
```

### 3) Configure environment

```bash
cp .env.example .env
nano .env
```

Set real values for:
- DB_NAME
- DB_USER
- DB_PASS
- GEMINI_API_KEY
- JINA_API_KEY
- FRONTEND_ORIGINS (your frontend domain)

Do not set `DB_HOST` in `.env` for compose deployment; compose forces it to `db` for the app container.

### 4) Start stack

```bash
docker compose up -d --build
```

### 5) Verify stack

```bash
docker compose ps
docker compose logs -f app
curl -i http://localhost:8000/health
```

### 6) Keep it running after reboot

Both services use `restart: unless-stopped`, so they restart automatically with Docker daemon after host reboot.

### 7) Update deploys

```bash
git pull
docker compose up -d --build
```

### 8) Optional reverse proxy + TLS

For production HTTPS, put Nginx or Caddy in front of port 8000 and terminate TLS there.

## 1) PostgreSQL Setup

1. Create database and user.
2. Grant privileges.
3. Enable pgvector extension.

Run these SQL commands as a database admin user:

```sql
CREATE USER rag_user WITH PASSWORD 'replace_with_strong_password';
CREATE DATABASE rag_db OWNER rag_user;
\c rag_db
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
```

Notes:
- If CREATE EXTENSION fails, install pgvector package on the DB host first.
- Ensure network/firewall allows backend host to reach PostgreSQL.

## 2) Backend Environment

1. Copy `.env.example` to `.env`.
2. Fill in real values for DB and API keys.

Required startup variables:
- DB_HOST
- DB_PORT
- DB_NAME
- DB_USER
- DB_PASS
- GEMINI_API_KEY
- JINA_API_KEY

The app now fails fast at startup if these are missing/invalid.

## 3) Windows Deployment (Waitress)

### 3.1 Install dependencies

From repository root:

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install waitress
```

### 3.2 Run backend

```powershell
waitress-serve --host=0.0.0.0 --port=8000 rag_server:app
```

### 3.3 Run as Windows Service (NSSM)

1. Install NSSM.
2. Create service pointing to Python executable.
3. Use module invocation so venv packages are used.

Example service command:

Program:
- C:\path\to\rag-chatbot\venv\Scripts\python.exe

Arguments:
- -m waitress --host=0.0.0.0 --port=8000 rag_server:app

Startup directory:
- C:\path\to\rag-chatbot

Set service startup type to Automatic.

## 4) Linux Deployment (systemd + gunicorn)

### 4.1 Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### 4.2 Start with gunicorn

```bash
gunicorn -w 2 -k gthread -b 0.0.0.0:8000 rag_server:app
```

### 4.3 Example systemd unit

Create `/etc/systemd/system/rag-chatbot.service`:

```ini
[Unit]
Description=RAG Chatbot Backend
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/rag-chatbot
EnvironmentFile=/opt/rag-chatbot/.env
ExecStart=/opt/rag-chatbot/venv/bin/gunicorn -w 2 -k gthread -b 0.0.0.0:8000 rag_server:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-chatbot
sudo systemctl start rag-chatbot
sudo systemctl status rag-chatbot
```

## 5) Docker Compose (Backend + PostgreSQL)

This repository now includes:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

Compose uses:
- `pgvector/pgvector:pg16` for PostgreSQL + pgvector
- `gunicorn` for Flask app serving

## 6) Reverse Proxy and TLS

Use Nginx or Caddy in front of backend:
- Terminate TLS
- Proxy to backend port 8000
- Restrict allowed origins via FRONTEND_ORIGINS

## 7) Post-deploy Smoke Checks

Health:

```bash
curl -i http://localhost:8000/health
```

CORS preflight:

```bash
curl -i -X OPTIONS http://localhost:8000/health \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET"
```

PDF download contract (GET + snake_case):

```bash
curl -i "http://localhost:8000/download_pdf?session_id=<session_id>&owner_key=<owner_key>&turn_index=1"
```

## 8) Production Checklist

- Use strong DB password and rotate periodically
- Restrict DB ingress to backend host(s) only
- Keep .env out of version control
- Set FRONTEND_ORIGINS to production frontend domain(s)
- Enable process restart (service manager)
- Set up centralized logs and monitoring
- Run backup policy for PostgreSQL
