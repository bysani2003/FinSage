# Running Stock Predictor Locally (Without Docker)

## Prerequisites

1. **Python 3.8+** - Download from [python.org](https://www.python.org/downloads/)
2. **PostgreSQL** - Download from [postgresql.org](https://www.postgresql.org/download/windows/)

## Setup Instructions

### 1. Install PostgreSQL

1. Download and install PostgreSQL for Windows
2. During installation, set password for `postgres` user (remember this!)
3. Default port should be `5432`

### 2. Create Database

Open **pgAdmin** or **psql** command line:

```sql
CREATE DATABASE stockdb;
```

Run the initialization script:
```bash
psql -U postgres -d stockdb -f init.sql
```

### 3. Setup Backend

```bash
cd backend
pip install -r requirements.txt

# Set environment variable (Windows CMD)
set DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/stockdb

# Or PowerShell
$env:DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/stockdb"

# Run backend
python app.py
```

Backend will run on: http://localhost:5000

### 4. Setup Dashboard

**First, update the backend URL:**

Open `dashboard/app.py` and change line 16:
```python
# BACKEND_URL = "http://backend:5000"  # Comment this
BACKEND_URL = "http://localhost:5000"   # Use this instead
```

**Then run dashboard:**
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Dashboard will run on: http://localhost:8501

## Quick Alternative: SQLite Version

If PostgreSQL setup is too complex, I can create a SQLite version that requires no database installation.
