#!/bin/bash
set -e  # Exit on any error

# Log all output
exec > >(tee -a /var/log/startup.log)
exec 2>&1

echo "=========================================="
echo "Starting Threadboard setup at $(date)"
echo "=========================================="

# ============================================================================
# SECTION 1: ENVIRONMENT CHECK
# ============================================================================
echo ""
echo "=== Environment Check ==="
echo "  - OS: $(uname -a)"
echo "  - User: $(whoami)"
echo "  - Current directory: $(pwd)"
echo "  - Docker version: $(docker --version)"
echo "  - Available disk space: $(df -h /)"
echo ""

# ============================================================================
# SECTION 2: CREATE BASE DIRECTORIES
# ============================================================================
echo "Creating base directories..."
sudo mkdir -p /var/lib/app /var/lib/toolbox
sudo chown -R $USER:$USER /var/lib/app /var/lib/toolbox
sudo chmod 755 /var/lib/app /var/lib/toolbox

# ============================================================================
# SECTION 3: WAIT FOR DOCKER
# ============================================================================
echo "Waiting for Docker to be ready..."
for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
        echo "Docker is ready"
        break
    fi
    echo "Waiting for Docker... ($i/30)"
    sleep 5
done

# ============================================================================
# SECTION 4: CONFIGURE DOCKER AUTHENTICATION
# ============================================================================
echo "Configuring Docker authentication for Artifact Registry..."

# Create a writable directory for Docker config (COS has read-only filesystem)
sudo mkdir -p /var/lib/docker-config
if [ "$(whoami)" = "root" ]; then
    sudo chown root:root /var/lib/docker-config
else
    sudo chown $USER:$USER /var/lib/docker-config
fi
sudo chmod 755 /var/lib/docker-config
export DOCKER_CONFIG=/var/lib/docker-config

# Get access token from metadata server
echo "Getting access token from metadata server..."
ACCESS_TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$ACCESS_TOKEN" ]; then
    echo "Failed to get access token from metadata server"
    exit 1
fi

# Configure Docker to use the access token
echo "Configuring Docker with access token..."
echo "$ACCESS_TOKEN" | docker login -u oauth2accesstoken --password-stdin https://${region}-docker.pkg.dev || {
    echo "Failed to authenticate Docker with Artifact Registry"
    exit 1
}

echo "Docker authentication configured successfully"

# ============================================================================
# SECTION 5: DOCKER COMPOSE CONFIGURATION
# ============================================================================
echo ""
echo "=== Creating Docker Compose Configuration ==="

cd /var/lib/app
cat > docker-compose.yml <<'COMPOSE'
version: '3.8'

services:
  threadboard:
    image: ${region}-docker.pkg.dev/${project_id}/threadboard/threadboard:latest
    container_name: threadboard
    ports:
      - "80:5000"
      - "443:5000"
    environment:
      - REDDIT_CLIENT_ID_SECRET=${reddit_client_id_secret}
      - REDDIT_CLIENT_SECRET_SECRET=${reddit_client_secret_secret}
      - GEMINI_API_KEY_SECRET=${gemini_api_key_secret}
      - SECRET_KEY_SECRET=${flask_secret_key_secret}
      - GOOGLE_CLOUD_PROJECT=${project_id}
      - USE_GEMINI=true
      - PORT=5000
    volumes:
      - /var/lib/app/data:/app/data
    restart: unless-stopped
    logging:
      driver: "gcplogs"
      options:
        gcp-project: "${project_id}"
        gcp-log-cmd: "true"
        labels: "service,container_name"
    labels:
      - "service=threadboard"
      - "container_name=threadboard"
COMPOSE

# ============================================================================
# SECTION 6: START APPLICATION
# ============================================================================
echo ""
echo "=== Starting Application ==="

cd /var/lib/app
export DOCKER_CONFIG=/var/lib/docker-config

# Install docker-compose
echo "Installing docker-compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /var/lib/toolbox/docker-compose
    chmod +x /var/lib/toolbox/docker-compose
    DOCKER_COMPOSE_CMD="/var/lib/toolbox/docker-compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# Verify docker-compose is working
if ! $DOCKER_COMPOSE_CMD --version; then
    echo "docker-compose installation failed. Exiting."
    exit 1
fi

# Pull images
echo "Pulling Docker images..."
if ! $DOCKER_COMPOSE_CMD pull; then
    echo "WARNING: Failed to pull Docker images. Attempting to continue with cached images..."
fi

echo "Starting Threadboard application..."
$DOCKER_COMPOSE_CMD up -d threadboard

# Wait for the application to be ready
echo "Waiting for application to be ready..."
for i in {1..60}; do
    if curl -f -s http://localhost:5000/ > /dev/null 2>&1; then
        echo "Application is ready!"
        break
    fi
    echo "Waiting for application... ($i/60)"
    sleep 5
done

# ============================================================================
# SECTION 7: CREATE MANAGEMENT SCRIPTS
# ============================================================================
echo ""
echo "=== Creating Management Scripts ==="

# Create update script
cat > /var/lib/toolbox/update-app <<'UPDATE_SCRIPT'
#!/bin/bash
set -e

echo "=========================================="
echo "Starting Threadboard application update..."
echo "=========================================="

cd /var/lib/app

# Fix Docker config directory permissions
export DOCKER_CONFIG=/var/lib/docker-config
sudo mkdir -p /var/lib/docker-config
if [ "$(whoami)" = "root" ]; then
    sudo chown root:root /var/lib/docker-config
else
    sudo chown $USER:$USER /var/lib/docker-config
fi
sudo chmod 755 /var/lib/docker-config

# Authenticate Docker with Artifact Registry
echo "Authenticating with Artifact Registry..."
ACCESS_TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$ACCESS_TOKEN" ]; then
    echo "Failed to get access token from metadata server"
    exit 1
fi

echo "$ACCESS_TOKEN" | docker login -u oauth2accesstoken --password-stdin https://${region}-docker.pkg.dev || {
    echo "Failed to authenticate Docker"
    exit 1
}

# Find docker-compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif [ -x "/var/lib/toolbox/docker-compose" ]; then
    COMPOSE_CMD="/var/lib/toolbox/docker-compose"
else
    echo "docker-compose not found. Please install it first."
    exit 1
fi

echo "Pulling latest images..."
$COMPOSE_CMD pull || {
    echo "Warning: Failed to pull latest images, continuing with existing..."
}

echo "Stopping and removing old containers..."
$COMPOSE_CMD down --remove-orphans

echo "Cleaning up unused Docker images and containers..."
docker system prune -f --filter until=2h || echo "Warning: Docker cleanup failed, continuing..."

echo "Starting Threadboard application..."
$COMPOSE_CMD up -d threadboard

echo "Waiting for services to be healthy..."
sleep 10

echo "Application status:"
$COMPOSE_CMD ps

echo ""
echo "✅ Threadboard application updated successfully!"
UPDATE_SCRIPT

chmod +x /var/lib/toolbox/update-app

# ============================================================================
# SECTION 8: SYSTEMD DEPLOYMENT MONITOR
# ============================================================================
echo ""
echo "=== Setting up systemd services ==="

# Create deployment monitor script
cat > /var/lib/toolbox/deployment-monitor.sh <<'DEPLOY_MONITOR'
#!/bin/bash
set -e

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HEADERS="Metadata-Flavor: Google"
LAST_SIGNAL_FILE="/var/lib/toolbox/last-deployment-signal"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [DEPLOY-MONITOR] $1"
}

get_metadata() {
    local key="$1"
    curl -s -f -H "$HEADERS" "$METADATA_URL/$key" 2>/dev/null || echo ""
}

check_deployment_signal() {
    local current_signal=$(get_metadata "deploy-signal")

    if [ -z "$current_signal" ]; then
        return 0
    fi

    local last_signal=""
    if [ -f "$LAST_SIGNAL_FILE" ]; then
        last_signal=$(cat "$LAST_SIGNAL_FILE")
    fi

    if [ "$current_signal" != "$last_signal" ]; then
        log "🔔 New deployment signal detected: $current_signal"
        echo "$current_signal" > "$LAST_SIGNAL_FILE"

        log "🚀 Starting deployment..."
        if /var/lib/toolbox/update-app; then
            log "✅ Deployment completed successfully"
        else
            log "❌ Deployment failed"
        fi
    fi
}

log "🚀 Deployment monitor started (checking every 30s)"

while true; do
    check_deployment_signal
    sleep 30
done
DEPLOY_MONITOR

chmod +x /var/lib/toolbox/deployment-monitor.sh

# Create systemd service for deployment monitoring
cat > /etc/systemd/system/threadboard-deployment-monitor.service <<'DEPLOY_SERVICE'
[Unit]
Description=Threadboard Deployment Monitor
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
ExecStart=/var/lib/toolbox/deployment-monitor.sh
Restart=always
RestartSec=10
User=root
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
DEPLOY_SERVICE

# Enable and start the service
echo "🔧 Enabling systemd services..."
systemctl daemon-reload
systemctl enable threadboard-deployment-monitor.service

echo "🚀 Starting systemd services..."
systemctl start threadboard-deployment-monitor.service

echo "✅ Systemd services configured and started"

# ============================================================================
# SECTION 9: FINAL STATUS
# ============================================================================
echo ""
echo "=== Startup Complete ==="

# Show container status
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    /var/lib/toolbox/docker-compose ps
fi

echo ""
echo "=========================================="
echo "Threadboard Setup Complete!"
echo "=========================================="
echo ""
echo "Application Information:"
echo "  - URL: http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H 'Metadata-Flavor: Google'):5000"
echo "  - Port: 5000"
echo ""
echo "Management Commands:"
echo "  - Update: sudo /var/lib/toolbox/update-app"
echo "  - Logs: docker logs threadboard"
echo "  - Status: docker-compose ps"
echo ""
echo "Setup completed at $(date)"
echo "=========================================="
