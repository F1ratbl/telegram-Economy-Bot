from flask import Blueprint, jsonify, request

from app.services.bot_service import start_background_update


bp = Blueprint("webhook", __name__)


@bp.get("/")
def root():
    return jsonify({"status": "running"})


@bp.route("/webhook", methods=["GET", "POST"])
@bp.route("/webhook/", methods=["GET", "POST"])
def telegram_webhook():
    if request.method == "GET":
        return jsonify({"status": "webhook-ready"})
    update = request.get_json(silent=True) or {}
    try:
        start_background_update(update)
    except Exception:
        pass
    return jsonify({"status": "ok"})
