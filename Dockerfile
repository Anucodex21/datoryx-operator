FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    fastapi "uvicorn[standard]" bcrypt pyjwt email-validator

COPY backend/ backend/
COPY frontend/ frontend/

WORKDIR /app/backend

ENV DATORYX_JWT_SECRET=change-me-in-production
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
