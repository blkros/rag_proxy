FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

 # (rag-proxy 이미지 빌드 단계에 추가)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng poppler-utils \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=120 -r /app/requirements.txt

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ocrmypdf qpdf ghostscript \
    tesseract-ocr tesseract-ocr-kor tesseract-ocr-script-hang \
    poppler-utils \
    fonts-nanum fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

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