``` bash
python-backend/
│
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── __init__.py
│
│   ├── api/                        # REST API layer (contracts)
│   │   └── v1/
│   │       ├── auth/
│   │       │   ├── router.py       # JWT, device trust
│   │       │   └── schemas.py
│   │
│   │       ├── users/
│   │       │   ├── router.py
│   │       │   └── schemas.py
│   │
│   │       ├── courses/
│   │       │   ├── router.py
│   │       │   └── schemas.py
│   │
│   │       ├── sessions/
│   │       │   ├── router.py       # session lifecycle
│   │       │   ├── qr_router.py    # QR generation / validation
│   │       │   └── schemas.py
│   │
│   │       ├── face/
│   │       │   ├── router.py       # enroll / verify
│   │       │   └── schemas.py
│   │
│   │       ├── attendance/
│   │       │   ├── router.py       # attendance flow
│   │       │   ├── geo_router.py   # geofence validation
│   │       │   └── schemas.py
│   │
│   │       └── system/
│   │           ├── router.py       # health, metrics
│   │           └── schemas.py
│
│   ├── core/                       # Pure business logic (framework-agnostic)
│   │   ├── face_engine/
│   │   │   ├── detector.py
│   │   │   ├── encoder.py
│   │   │   ├── matcher.py          # pgvector similarity
│   │   │   └── validator.py
│   │   │
│   │   ├── qr_engine/
│   │   │   ├── token_generator.py
│   │   │   ├── token_validator.py
│   │   │   └── expiry_rules.py
│   │   │
│   │   ├── geo_engine/
│   │   │   ├── geofence_validator.py
│   │   │   └── location_utils.py
│   │   │
│   │   └── attendance_engine/
│   │       ├── state_machine.py
│   │       └── rules.py
│
│   ├── services/                   # External systems (I/O)
│   │   └── postgres/
│   │       ├── session.py          # async DB session
│   │       │
│   │       ├── repositories/
│   │       │   ├── user_repository.py
│   │       │   ├── course_repository.py
│   │       │   ├── session_repository.py
│   │       │   ├── qr_token_repository.py
│   │       │   ├── face_template_repository.py
│   │       │   ├── attendance_repository.py
│   │       │   └── liveness_repository.py
│   │       │
│   │       └── migrations/
│   │           ├── enable_pgvector.sql
│   │           ├── enable_postgis.sql
│   │           └── create_tables.sql
│
│   ├── workers/                    # Async / background processing
│   │   ├── face_tasks/
│   │   ├── qr_cleanup_tasks/
│   │   └── geo_validation_tasks/
│
│   ├── config/
│   │   ├── settings.py             # env, thresholds, expiry
│   │   └── logging.py
│
│   ├── utils/
│   │   ├── image_utils.py
│   │   ├── crypto_utils.py
│   │   ├── geo_utils.py
│   │   └── time_utils.py
│
│   └── dependencies/
│       ├── auth.py                 # FastAPI dependency injection
│       └── db.py
│
├── tests/
│   ├── unit/
│   │   ├── face_engine/
│   │   ├── qr_engine/
│   │   └── geo_engine/
│   │
│   ├── integration/
│   │   ├── test_pgvector.py
│   │   ├── test_postgis.py
│   │   ├── test_qr_flow.py
│   │   └── test_attendance_flow.py
│   │
│   └── contract/
│       └── test_api_contracts.py
│
├── deployments/
│   ├── local/
│   │   └── docker-compose.yml
│   │
│   ├── k8s/
│   │   ├── namespace.yaml
│   │   ├── backend/
│   │   ├── postgres/
│   │   └── ingress/
│   │
│   └── ci/
│       └── github-actions.yml
│
├── Dockerfile
├── requirements.txt
├── .env
├── .dockerignore
├── README.md
└── logs/
    └── backend.log

```