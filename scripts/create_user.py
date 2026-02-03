#!/usr/bin/env python3
"""
Create a new user with a secure password.

Usage:
    uv run python scripts/create_user.py user@example.com
    uv run python scripts/create_user.py user@example.com --name "John Doe"
    uv run python scripts/create_user.py user@example.com --admin
    uv run python scripts/create_user.py user@example.com --password "custom-password"
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.database import init_pool, get_cursor
from webapp.auth import create_user, generate_password, hash_password


def main():
    parser = argparse.ArgumentParser(description="Create a new user")
    parser.add_argument("email", help="User email address")
    parser.add_argument("--name", "-n", help="User's display name")
    parser.add_argument("--admin", "-a", action="store_true", help="Make user an admin")
    parser.add_argument("--password", "-p", help="Set a specific password (otherwise generates one)")

    args = parser.parse_args()

    # Initialize database
    init_pool()

    # Check if user already exists
    with get_cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE email = %s", (args.email.lower(),))
        if cursor.fetchone():
            print(f"Error: User with email '{args.email}' already exists.")
            sys.exit(1)

    # Generate or use provided password
    password = args.password or generate_password()

    # Create user
    user_id = create_user(
        email=args.email.lower(),
        password=password,
        name=args.name,
        is_admin=args.admin
    )

    print(f"\nUser created successfully!")
    print(f"  ID:    {user_id}")
    print(f"  Email: {args.email.lower()}")
    if args.name:
        print(f"  Name:  {args.name}")
    print(f"  Admin: {'Yes' if args.admin else 'No'}")
    print(f"\n  Password: {password}")
    print(f"\n  (Save this password - it cannot be recovered)")


if __name__ == "__main__":
    main()
