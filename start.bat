@echo off

REM Start PostgreSQL service
net start postgresql-x64-15

REM Connection parameters
set PGHOST=localhost
set PGPORT=5432
set PGDATABASE=attendance_db
set PGUSER=postgres
set PGPASSWORD=root

REM Path to psql
set PSQL_PATH="C:\Program Files\PostgreSQL\18\bin\psql.exe"

REM Connect
%PSQL_PATH%

pause
