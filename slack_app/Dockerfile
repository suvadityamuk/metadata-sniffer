FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PYTHONUNBUFFERED=1
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
CMD ["python", "-m", "slack_app.app"]
