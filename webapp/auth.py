"""
Authentication module for the webapp.

Provides user management, login/logout, and route protection.
"""

import os
import secrets
from datetime import datetime
from functools import wraps
from typing import Optional

import bcrypt
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

from eval.database import get_cursor

# Blueprint for auth routes
auth_bp = Blueprint('auth', __name__)

# Login manager instance (initialized in init_auth)
login_manager = LoginManager()


class User(UserMixin):
    """User model for Flask-Login."""

    def __init__(self, id: int, email: str, name: Optional[str], is_active: bool, is_admin: bool):
        self.id = id
        self.email = email
        self.name = name
        self._is_active = is_active
        self.is_admin = is_admin

    @property
    def is_active(self):
        return self._is_active

    @staticmethod
    def get_by_id(user_id: int) -> Optional['User']:
        """Load user by ID."""
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT id, email, name, is_active, is_admin FROM users WHERE id = %s",
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                return User(
                    id=row['id'],
                    email=row['email'],
                    name=row['name'],
                    is_active=row['is_active'],
                    is_admin=row['is_admin']
                )
        return None

    @staticmethod
    def get_by_email(email: str) -> Optional['User']:
        """Load user by email."""
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT id, email, name, is_active, is_admin FROM users WHERE email = %s",
                (email,)
            )
            row = cursor.fetchone()
            if row:
                return User(
                    id=row['id'],
                    email=row['email'],
                    name=row['name'],
                    is_active=row['is_active'],
                    is_admin=row['is_admin']
                )
        return None

    @staticmethod
    def verify_password(email: str, password: str) -> Optional['User']:
        """Verify password and return user if valid."""
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT id, email, name, password_hash, is_active, is_admin FROM users WHERE email = %s",
                (email,)
            )
            row = cursor.fetchone()
            if row and bcrypt.checkpw(password.encode('utf-8'), row['password_hash'].encode('utf-8')):
                # Update last login time
                cursor.execute(
                    "UPDATE users SET last_login_at = %s WHERE id = %s",
                    (datetime.now(), row['id'])
                )
                return User(
                    id=row['id'],
                    email=row['email'],
                    name=row['name'],
                    is_active=row['is_active'],
                    is_admin=row['is_admin']
                )
        return None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def generate_password(length: int = 16) -> str:
    """Generate a secure random password."""
    return secrets.token_urlsafe(length)


def create_user(email: str, password: str, name: Optional[str] = None, is_admin: bool = False) -> int:
    """Create a new user. Returns the user ID."""
    password_hash = hash_password(password)
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO users (email, password_hash, name, is_admin)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (email, password_hash, name, is_admin)
        )
        return cursor.fetchone()['id']


def init_auth(app):
    """Initialize authentication for the Flask app."""
    # Set secret key for sessions
    app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

    # Configure login manager
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.get_by_id(int(user_id))

    # Register blueprint
    app.register_blueprint(auth_bp)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)

        user = User.verify_password(email, password)
        if user:
            if not user.is_active:
                flash('Your account has been deactivated.', 'error')
                return render_template('auth/login.html')

            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout and redirect to login page."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
