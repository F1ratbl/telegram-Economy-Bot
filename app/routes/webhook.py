from flask import Blueprint, jsonify, request

from app.services.bot_service import start_background_update


bp = Blueprint("webhook", __name__)


@bp.get("/")
def root():
    return jsonify({"status": "running"})


@bp.post("/webhook")
def telegram_webhook():
    update = request.get_json(silent=True) or {}
    try:
        start_background_update(update)
    except Exception:
        pass
    return jsonify({"status": "ok"})
