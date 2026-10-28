#!/usr/bin/env bash
set -euo pipefail

create_database() {
    local db_name="$1"
    echo ">>> Création de la base '${db_name}' (si absente)"
    psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname postgres <<-EOSQL
        SELECT 'CREATE DATABASE "${db_name}"'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${db_name}')\gexec
EOSQL
}

create_database "${AIRFLOW_DB_NAME:-airflow}"
create_database "${MLFLOW_DB_NAME:-mlflow}"

echo ">>> Bases de données BrainScan-v2 prêtes."
