FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 소스는 명시적으로 복사 (api/, src/가 반드시 컨텍스트 최상위에 있어야 함)
COPY api/ /app/api
COPY src/ /app/src
COPY data/ /app/data

# 패키지 인식 보강 (없으면 만들어 줌)
RUN [ -f /app/api/__init__.py ] || touch /app/api/__init__.py; \
    [ -f /app/src/__init__.py ] || touch /app/src/__init__.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]