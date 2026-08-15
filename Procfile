web: gunicorn dashboard:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
worker: python3 -c "from ai_agent_daemon import run_daemon; run_daemon(generations=None)"
