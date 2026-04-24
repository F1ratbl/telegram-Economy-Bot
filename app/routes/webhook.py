import logging

from flask import Blueprint, jsonify, request

from app.services.bot_service import start_background_update


bp = Blueprint("webhook", __name__)
logger = logging.getLogger("economy-assistant-bot")


@bp.get("/")
def root():
    return jsonify({"status": "running"})


@bp.route("/webhook", methods=["GET", "POST"])
@bp.route("/webhook/", methods=["GET", "POST"])
def telegram_webhook():
    if request.method == "GET":
        return jsonify({"status": "webhook-ready"})
    update = request.get_json(silent=True) or {}
    logger.info("Webhook update alindi. Top-level anahtarlar: %s", list(update.keys()))
    try:
        start_background_update(update)
    except Exception:
        logger.exception("Webhook update islenirken route seviyesinde hata olustu.")
    return jsonify({"status": "ok"})
