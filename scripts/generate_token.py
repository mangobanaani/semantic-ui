#!/usr/bin/env python3
"""Generate JWT token for authentication.

Usage:
    python scripts/generate_token.py --email user@example.com --name "User Name"
    python scripts/generate_token.py --email user@example.com --name "User Name" --hours 48
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    import jwt
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install PyJWT python-dotenv")
    sys.exit(1)


def generate_token(email: str, name: str, hours: int = 24) -> str:
    """Generate a JWT token.

    Args:
        email: User email
        name: User name
        hours: Token expiration in hours

    Returns:
        JWT token string
    """
    # Load environment variables
    load_dotenv()

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret or jwt_secret == "your-secret-key-change-this-in-production":
        print("ERROR: JWT_SECRET not configured in .env file")
        print("Please set a secure JWT_SECRET in your .env file")
        sys.exit(1)

    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")

    # Create payload
    payload = {
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(hours=hours),
        "iat": datetime.utcnow(),
    }

    # Generate token
    token = jwt.encode(payload, jwt_secret, algorithm=jwt_algorithm)

    return token


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate JWT token for Semantic Kernel UI authentication"
    )
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--name", required=True, help="User full name")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Token expiration in hours (default: 24)",
    )

    args = parser.parse_args()

    try:
        token = generate_token(args.email, args.name, args.hours)

        print("\n" + "=" * 80)
        print("JWT Token Generated Successfully")
        print("=" * 80)
        print(f"\nUser: {args.name}")
        print(f"Email: {args.email}")
        print(f"Expires in: {args.hours} hours")
        print(f"\nToken:\n{token}")
        print("\n" + "=" * 80)
        print("\nTo use this token:")
        print("1. Copy the token above")
        print("2. Go to the login page")
        print("3. Paste the token in the 'Enter your JWT token' field")
        print("4. Click 'Login with Token'")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nERROR: Failed to generate token: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
