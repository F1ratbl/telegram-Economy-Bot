from flask import Flask

from app.core.logging import setup_logging
from app.routes import bp
from app.services.bot_service import initialize_webhook


def create_app() -> Flask:
    setup_logging()
    app = Flask(__name__)
    app.register_blueprint(bp)
    initialize_webhook()
    return app
