#!/bin/bash
echo "Restoring database from backup.sql..."
docker exec -i kiavi_postgres psql -U postgres -d kiavidb < backup.sql
echo "✓ Database restored successfully!"
