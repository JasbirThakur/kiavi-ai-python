#!/bin/bash
set -e
echo "Restoring database from backup.sql..."

DOCKER_CMD="docker"
if ! docker ps > /dev/null 2>&1; then
    if sudo docker ps > /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    fi
fi

$DOCKER_CMD exec -i kiavi_postgres psql -U postgres -d kiavidb < backup.sql
echo "✓ Database restored successfully!"
