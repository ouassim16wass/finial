# Utiliser une image Python légère
FROM python:3.9-slim

WORKDIR /app

# Installer PostgreSQL et ses dépendances système pour `pg_config`
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Vérifier si `pg_config` est bien installé dans le conteneur
RUN which pg_config

# Copier les fichiers requirements.txt
COPY requirements.txt requirements.txt

# Installer `psycopg2-binary` en premier pour éviter les conflits
RUN pip install --no-cache-dir psycopg2-binary

# Installer les autres dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet après installation des dépendances
COPY . .

EXPOSE 5000

# Lancer Flask avec Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
