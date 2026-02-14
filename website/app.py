import os
import sys
from pathlib import Path
from flask import Flask

from config import Config
from extensions import db, login_manager


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')).parent, exist_ok=True)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)

    # Register blueprints
    from routes.auth import auth_bp
    from routes.main import main_bp
    from routes.search import search_bp
    from routes.upload import upload_bp
    from routes.collection import collection_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(collection_bp)

    # Create database tables & register user_loader
    with app.app_context():
        from models.user import User
        db.create_all()

        @login_manager.user_loader
        def load_user(user_id):
            return User.query.get(int(user_id))

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=36368)
