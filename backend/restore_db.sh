#!/bin/bash
set -e
echo "Restoring database from backup.sql..."

DOCKER_CMD="docker"
if ! docker ps > /dev/null 2>&1; then
    if sudo docker ps > /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    fi
fi

# Find running postgres container (checks both standalone backend and root setup)
CONTAINER_NAME="kiavi_backend_postgres"
if ! $DOCKER_CMD ps --format '{{.Names}}' | grep -q "^kiavi_backend_postgres$"; then
    if $DOCKER_CMD ps --format '{{.Names}}' | grep -q "^kiavi_postgres$"; then
        CONTAINER_NAME="kiavi_postgres"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Target database container: $CONTAINER_NAME"
$DOCKER_CMD exec -i "$CONTAINER_NAME" psql -U postgres -d kiavidb < "$SCRIPT_DIR/backup.sql"
echo "✓ Database restored successfully!"

