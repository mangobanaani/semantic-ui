"""Authentication manager for OAuth and JWT token authentication."""
from __future__ import annotations

import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
import streamlit as st
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class AuthConfig(BaseSettings):
    """Authentication configuration."""

    # Enable/disable authentication
    enable_auth: bool = Field(default=False, validation_alias="ENABLE_AUTH")

    # JWT Settings
    jwt_secret: str = Field(default="", validation_alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(
        default=24, validation_alias="JWT_EXPIRATION_HOURS"
    )

    # OAuth Google Settings
    enable_google_oauth: bool = Field(
        default=False, validation_alias="ENABLE_GOOGLE_OAUTH"
    )
    google_client_id: str = Field(default="", validation_alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(
        default="", validation_alias="GOOGLE_CLIENT_SECRET"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:8501", validation_alias="GOOGLE_REDIRECT_URI"
    )

    # OAuth Facebook Settings
    enable_facebook_oauth: bool = Field(
        default=False, validation_alias="ENABLE_FACEBOOK_OAUTH"
    )
    facebook_app_id: str = Field(default="", validation_alias="FACEBOOK_APP_ID")
    facebook_app_secret: str = Field(
        default="", validation_alias="FACEBOOK_APP_SECRET"
    )
    facebook_redirect_uri: str = Field(
        default="http://localhost:8501", validation_alias="FACEBOOK_REDIRECT_URI"
    )

    # Allowed users (optional - restrict to specific emails)
    allowed_users: str = Field(
        default="", validation_alias="ALLOWED_USERS"
    )  # Comma-separated emails

    # Session Settings
    session_cookie_name: str = Field(
        default="sk_session", validation_alias="SESSION_COOKIE_NAME"
    )
    session_expiration_hours: int = Field(
        default=24, validation_alias="SESSION_EXPIRATION_HOURS"
    )

    # Security Settings
    enforce_https: bool = Field(default=True, validation_alias="ENFORCE_HTTPS")
    max_login_attempts: int = Field(default=5, validation_alias="MAX_LOGIN_ATTEMPTS")
    login_attempt_window_minutes: int = Field(
        default=15, validation_alias="LOGIN_ATTEMPT_WINDOW_MINUTES"
    )

    # MFA Settings (optional)
    enable_mfa: bool = Field(default=False, validation_alias="ENABLE_MFA")
    mfa_issuer_name: str = Field(
        default="Semantic Kernel UI", validation_alias="MFA_ISSUER_NAME"
    )

    # Audit Logging
    enable_audit_log: bool = Field(default=True, validation_alias="ENABLE_AUDIT_LOG")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class User(BaseModel):
    """User model."""

    email: str
    name: str
    picture: Optional[str] = None
    provider: str  # "google", "facebook", "jwt"
    authenticated_at: datetime
    mfa_verified: bool = False
    session_token: Optional[str] = None


class AuthManager:
    """Manages authentication for the application."""

    def __init__(self, config: Optional[AuthConfig] = None):
        """Initialize auth manager.

        Args:
            config: Authentication configuration
        """
        self.config = config or AuthConfig()
        self._login_attempts: Dict[str, list[float]] = {}  # email -> timestamps
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize session state for authentication."""
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False
        if "user" not in st.session_state:
            st.session_state["user"] = None

    def is_authenticated(self) -> bool:
        """Check if user is authenticated.

        Returns:
            True if user is authenticated
        """
        if not self.config.enable_auth:
            return True

        return st.session_state.get("authenticated", False)

    def get_current_user(self) -> Optional[User]:
        """Get current authenticated user.

        Returns:
            Current user or None
        """
        return st.session_state.get("user")

    def create_jwt_token(self, email: str, name: str) -> str:
        """Create a JWT token for a user.

        Args:
            email: User email
            name: User name

        Returns:
            JWT token string
        """
        if not self.config.jwt_secret:
            raise ValueError("JWT_SECRET is not configured")

        payload = {
            "email": email,
            "name": name,
            "exp": datetime.now(timezone.utc)
            + timedelta(hours=self.config.jwt_expiration_hours),
            "iat": datetime.now(timezone.utc),
        }

        token = jwt.encode(
            payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm
        )
        return token

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload or None if invalid
        """
        if not self.config.jwt_secret:
            return None

        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def authenticate_with_jwt(self, token: str) -> tuple[bool, Optional[str]]:
        """Authenticate user with JWT token.

        Args:
            token: JWT token string

        Returns:
            (success, error_message)
        """
        # Verify token first to get email for rate limiting
        payload = self.verify_jwt_token(token)
        if not payload:
            self._audit_log("jwt_login", "unknown", False, "invalid_token")
            return False, "Authentication failed"

        email = payload.get("email", "")
        name = payload.get("name", "")

        # Check rate limit
        is_allowed, rate_limit_msg = self._check_rate_limit(email)
        if not is_allowed:
            self._audit_log("jwt_login", email, False, "rate_limited")
            return False, rate_limit_msg

        # Record attempt
        self._record_login_attempt(email)

        # Check if user is allowed
        if not self._is_user_allowed(email):
            self._audit_log("jwt_login", email, False, "user_not_allowed")
            return False, "Authentication failed"

        # Create user and set session
        user = User(
            email=email,
            name=name,
            provider="jwt",
            authenticated_at=datetime.now(timezone.utc),
            mfa_verified=not self.config.enable_mfa,  # MFA handled separately for JWT
        )

        # Check MFA
        if self.config.enable_mfa and not user.mfa_verified:
            st.session_state["pending_mfa_user"] = user
            self._audit_log("jwt_login", email, True, "pending_mfa")
            return True, None

        # Create session token
        session_token = self.create_session_token(user)
        user.session_token = session_token

        st.session_state["authenticated"] = True
        st.session_state["user"] = user

        self._audit_log("jwt_login", email, True, "success")
        return True, None

    def authenticate_with_google(self, auth_code: str) -> bool:
        """Authenticate user with Google OAuth.

        Args:
            auth_code: OAuth authorization code

        Returns:
            True if authentication successful
        """
        if not self.config.enable_google_oauth:
            return False

        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests

            # Verify the authorization code and get user info
            # This is a simplified version - in production you'd exchange
            # the code for tokens using google-auth library
            idinfo = id_token.verify_oauth2_token(
                auth_code, requests.Request(), self.config.google_client_id
            )

            email = idinfo.get("email", "")
            name = idinfo.get("name", "")
            picture = idinfo.get("picture")

            # Check if user is allowed
            if not self._is_user_allowed(email):
                return False

            # Create user and set session
            user = User(
                email=email,
                name=name,
                picture=picture,
                provider="google",
                authenticated_at=datetime.now(timezone.utc),
            )

            st.session_state["authenticated"] = True
            st.session_state["user"] = user

            return True

        except Exception:
            return False

    def authenticate_with_facebook(self, access_token: str) -> bool:
        """Authenticate user with Facebook OAuth.

        Args:
            access_token: Facebook access token

        Returns:
            True if authentication successful
        """
        if not self.config.enable_facebook_oauth:
            return False

        try:
            import requests

            # Verify access token and get user info
            response = requests.get(
                "https://graph.facebook.com/me",
                params={
                    "fields": "id,name,email,picture",
                    "access_token": access_token,
                },
                timeout=10,
            )

            if response.status_code != 200:
                return False

            user_data = response.json()
            email = user_data.get("email", "")
            name = user_data.get("name", "")
            picture = user_data.get("picture", {}).get("data", {}).get("url")

            # Check if user is allowed
            if not self._is_user_allowed(email):
                return False

            # Create user and set session
            user = User(
                email=email,
                name=name,
                picture=picture,
                provider="facebook",
                authenticated_at=datetime.now(timezone.utc),
            )

            st.session_state["authenticated"] = True
            st.session_state["user"] = user

            return True

        except Exception:
            return False

    def logout(self) -> None:
        """Logout current user."""
        user = st.session_state.get("user")
        if user:
            self._audit_log("logout", user.email, True, user.provider)

        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state.pop("pending_mfa_user", None)
        st.session_state.pop("mfa_secret", None)
        st.session_state.pop("mfa_uri", None)
        st.session_state.pop("user_mfa_secret", None)

        # Clear OAuth session if needed
        st.query_params.clear()

    def _is_user_allowed(self, email: str) -> bool:
        """Check if user email is allowed.

        Args:
            email: User email

        Returns:
            True if user is allowed
        """
        if not self.config.allowed_users:
            # No restrictions - all users allowed
            return True

        allowed_emails = [
            e.strip() for e in self.config.allowed_users.split(",") if e.strip()
        ]
        return email.lower() in [e.lower() for e in allowed_emails]

    def get_google_oauth_url(self) -> str:
        """Get Google OAuth authorization URL.

        Returns:
            OAuth URL
        """
        if not self.config.enable_google_oauth:
            return ""

        from urllib.parse import urlencode

        params = {
            "client_id": self.config.google_client_id,
            "redirect_uri": self.config.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
        }

        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    def get_facebook_oauth_url(self) -> str:
        """Get Facebook OAuth authorization URL.

        Returns:
            OAuth URL
        """
        if not self.config.enable_facebook_oauth:
            return ""

        from urllib.parse import urlencode

        params = {
            "client_id": self.config.facebook_app_id,
            "redirect_uri": self.config.facebook_redirect_uri,
            "scope": "email,public_profile",
        }

        return f"https://www.facebook.com/v12.0/dialog/oauth?{urlencode(params)}"

    def render_login_page(self) -> None:
        """Render login page in Streamlit."""
        st.title("Authentication Required")

        # Check for OAuth callback
        callback_error = self.handle_oauth_callback()
        if callback_error:
            st.error(callback_error)
        elif st.session_state.get("authenticated"):
            # Successfully authenticated via OAuth callback
            st.success("Authentication successful!")
            st.rerun()

        # Check for pending MFA
        if "pending_mfa_user" in st.session_state:
            user = st.session_state["pending_mfa_user"]
            self.render_mfa_setup(user)
            return

        # Check HTTPS enforcement
        is_https, https_error = self._check_https()
        if not is_https:
            st.error(https_error)
            return

        # JWT Token Login
        if self.config.jwt_secret:
            st.header("Login with Token")
            token = st.text_input(
                "Enter your JWT token:",
                type="password",
                key="jwt_token_input",
            )
            if st.button("Login with Token"):
                if token:
                    success, error_msg = self.authenticate_with_jwt(token)
                    if success:
                        if "pending_mfa_user" not in st.session_state:
                            st.success("Authentication successful!")
                            st.rerun()
                    else:
                        st.error(error_msg or "Authentication failed")
                else:
                    st.warning("Please enter a token")

        # OAuth Login Buttons
        oauth_enabled = (
            self.config.enable_google_oauth or self.config.enable_facebook_oauth
        )

        if oauth_enabled:
            st.header("Or login with OAuth")

            col1, col2 = st.columns(2)

            with col1:
                if self.config.enable_google_oauth:
                    google_url = self.get_google_oauth_url()
                    st.markdown(
                        f'<a href="{google_url}" target="_self">'
                        '<button style="width: 100%; padding: 10px; background-color: #4285F4; '
                        'color: white; border: none; border-radius: 4px; cursor: pointer;">'
                        "Sign in with Google</button></a>",
                        unsafe_allow_html=True,
                    )

            with col2:
                if self.config.enable_facebook_oauth:
                    facebook_url = self.get_facebook_oauth_url()
                    st.markdown(
                        f'<a href="{facebook_url}" target="_self">'
                        '<button style="width: 100%; padding: 10px; background-color: #1877F2; '
                        'color: white; border: none; border-radius: 4px; cursor: pointer;">'
                        "Sign in with Facebook</button></a>",
                        unsafe_allow_html=True,
                    )

        # Token Generation Info
        if self.config.jwt_secret:
            with st.expander("Need a token?"):
                st.info(
                    "Contact your administrator to generate a JWT token for you. "
                    "Tokens expire after "
                    f"{self.config.jwt_expiration_hours} hours."
                )

    # Rate Limiting
    def _check_rate_limit(self, identifier: str) -> tuple[bool, Optional[str]]:
        """Check if identifier is rate limited.

        Args:
            identifier: Email or IP to check

        Returns:
            (is_allowed, error_message)
        """
        current_time = time.time()
        window_seconds = self.config.login_attempt_window_minutes * 60

        # Clean old attempts
        if identifier in self._login_attempts:
            self._login_attempts[identifier] = [
                t
                for t in self._login_attempts[identifier]
                if current_time - t < window_seconds
            ]

        # Check if rate limited
        attempts = len(self._login_attempts.get(identifier, []))
        if attempts >= self.config.max_login_attempts:
            remaining_time = int(
                window_seconds - (current_time - self._login_attempts[identifier][0])
            )
            return False, f"Too many login attempts. Try again in {remaining_time // 60} minutes."

        return True, None

    def _record_login_attempt(self, identifier: str) -> None:
        """Record a login attempt.

        Args:
            identifier: Email or IP to record
        """
        if identifier not in self._login_attempts:
            self._login_attempts[identifier] = []
        self._login_attempts[identifier].append(time.time())

    # Audit Logging
    def _audit_log(self, event: str, user_email: str, success: bool, details: Optional[str] = None) -> None:
        """Log authentication event.

        Args:
            event: Event type (login, logout, mfa, etc.)
            user_email: User email
            success: Whether event was successful
            details: Additional details
        """
        if not self.config.enable_audit_log:
            return

        log_message = f"AUTH [{event}] user={user_email} success={success}"
        if details:
            log_message += f" details={details}"

        if success:
            logger.info(log_message)
        else:
            logger.warning(log_message)

    # HTTPS Enforcement
    def _check_https(self) -> tuple[bool, Optional[str]]:
        """Check if HTTPS is enforced.

        Returns:
            (is_https, error_message)
        """
        if not self.config.enforce_https:
            return True, None

        # Check if OAuth is enabled
        oauth_enabled = self.config.enable_google_oauth or self.config.enable_facebook_oauth

        if oauth_enabled:
            # In Streamlit, check the query params for scheme
            try:
                import streamlit.web.server.server as server
                if hasattr(server, 'Server') and hasattr(server.Server, '_server'):
                    # Production check would go here
                    # For now, allow localhost for development
                    if "localhost" not in self.config.google_redirect_uri and "localhost" not in self.config.facebook_redirect_uri:
                        if not self.config.google_redirect_uri.startswith("https://") and not self.config.facebook_redirect_uri.startswith("https://"):
                            return False, "OAuth requires HTTPS in production. Update redirect URIs to use https://"
            except Exception:
                pass  # Allow if we can't determine

        return True, None

    # Session Management
    def create_session_token(self, user: User) -> str:
        """Create a session token for user.

        Args:
            user: User to create session for

        Returns:
            Session token
        """
        session_data = {
            "email": user.email,
            "name": user.name,
            "provider": user.provider,
            "mfa_verified": user.mfa_verified,
            "exp": datetime.now(timezone.utc) + timedelta(hours=self.config.session_expiration_hours),
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(32),  # Unique session ID
        }

        if not self.config.jwt_secret:
            # Generate a random session token if no JWT secret
            return secrets.token_urlsafe(64)

        return jwt.encode(session_data, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify session token.

        Args:
            token: Session token

        Returns:
            Session data or None
        """
        if not self.config.jwt_secret:
            return None

        try:
            return jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def refresh_token(self, old_token: str) -> Optional[str]:
        """Refresh an expired or expiring token.

        Args:
            old_token: Current token

        Returns:
            New token or None
        """
        session_data = self.verify_session_token(old_token)
        if not session_data:
            return None

        # Create new token with extended expiration
        user = User(
            email=session_data["email"],
            name=session_data["name"],
            provider=session_data["provider"],
            authenticated_at=datetime.now(timezone.utc),
            mfa_verified=session_data.get("mfa_verified", False),
        )

        return self.create_session_token(user)

    # OAuth Callback Handling
    def handle_oauth_callback(self) -> Optional[str]:
        """Handle OAuth callback from query parameters.

        Returns:
            Error message or None if successful
        """
        try:
            # Get query parameters
            query_params = st.query_params

            # Check for OAuth code (Google)
            if "code" in query_params and self.config.enable_google_oauth:
                code_param = query_params["code"]
                code = code_param[0] if isinstance(code_param, list) else code_param  # type: ignore[unreachable,misc]

                # Exchange code for tokens
                success = self._exchange_google_code(code)
                if success:
                    # Clear query params
                    st.query_params.clear()
                    self._audit_log("oauth_login", st.session_state.get("user", {}).get("email", "unknown"), True, "google")
                    return None
                else:
                    self._audit_log("oauth_login", "unknown", False, "google")
                    return "Failed to authenticate with Google"

            # Check for Facebook OAuth
            if "code" in query_params and self.config.enable_facebook_oauth:
                code_param = query_params["code"]
                code = code_param[0] if isinstance(code_param, list) else code_param  # type: ignore[unreachable,misc]

                success = self._exchange_facebook_code(code)
                if success:
                    st.query_params.clear()
                    self._audit_log("oauth_login", st.session_state.get("user", {}).get("email", "unknown"), True, "facebook")
                    return None
                else:
                    self._audit_log("oauth_login", "unknown", False, "facebook")
                    return "Failed to authenticate with Facebook"

            # Check for error
            if "error" in query_params:
                error_param = query_params["error"]
                error = error_param[0] if isinstance(error_param, list) else error_param  # type: ignore[unreachable,misc]
                return f"Authentication error: {error}"

        except Exception:
            logger.exception("OAuth callback error")
            return "Authentication failed"

        return None

    def _exchange_google_code(self, code: str) -> bool:
        """Exchange Google authorization code for tokens.

        Args:
            code: Authorization code

        Returns:
            True if successful
        """
        try:
            from google_auth_oauthlib.flow import Flow

            # Create flow
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.config.google_client_id,
                        "client_secret": self.config.google_client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [self.config.google_redirect_uri],
                    }
                },
                scopes=["openid", "email", "profile"],
            )
            flow.redirect_uri = self.config.google_redirect_uri

            # Exchange code for tokens
            flow.fetch_token(code=code)

            # Get user info
            credentials = flow.credentials
            from google.oauth2 import id_token
            from google.auth.transport import requests

            idinfo = id_token.verify_oauth2_token(
                credentials.id_token, requests.Request(), self.config.google_client_id
            )

            email = idinfo.get("email", "")
            name = idinfo.get("name", "")
            picture = idinfo.get("picture")

            # Check if user is allowed
            if not self._is_user_allowed(email):
                return False

            # Create user
            user = User(
                email=email,
                name=name,
                picture=picture,
                provider="google",
                authenticated_at=datetime.now(timezone.utc),
                mfa_verified=not self.config.enable_mfa,  # Skip MFA for OAuth if not enabled
            )

            # Check MFA
            if self.config.enable_mfa and not user.mfa_verified:
                st.session_state["pending_mfa_user"] = user
                return True

            # Create session
            session_token = self.create_session_token(user)
            user.session_token = session_token

            st.session_state["authenticated"] = True
            st.session_state["user"] = user

            return True

        except Exception as e:
            logger.exception(f"Google OAuth error: {e}")
            return False

    def _exchange_facebook_code(self, code: str) -> bool:
        """Exchange Facebook authorization code for tokens.

        Args:
            code: Authorization code

        Returns:
            True if successful
        """
        try:
            import requests

            # Exchange code for access token
            token_url = "https://graph.facebook.com/v12.0/oauth/access_token"
            params = {
                "client_id": self.config.facebook_app_id,
                "client_secret": self.config.facebook_app_secret,
                "redirect_uri": self.config.facebook_redirect_uri,
                "code": code,
            }

            response = requests.get(token_url, params=params, timeout=10)
            if response.status_code != 200:
                return False

            access_token = response.json().get("access_token")
            if not access_token:
                return False

            # Get user info
            user_response = requests.get(
                "https://graph.facebook.com/me",
                params={"fields": "id,name,email,picture", "access_token": access_token},
                timeout=10,
            )

            if user_response.status_code != 200:
                return False

            user_data = user_response.json()
            email = user_data.get("email", "")
            name = user_data.get("name", "")
            picture = user_data.get("picture", {}).get("data", {}).get("url")

            # Check if user is allowed
            if not self._is_user_allowed(email):
                return False

            # Create user
            user = User(
                email=email,
                name=name,
                picture=picture,
                provider="facebook",
                authenticated_at=datetime.now(timezone.utc),
                mfa_verified=not self.config.enable_mfa,
            )

            # Check MFA
            if self.config.enable_mfa and not user.mfa_verified:
                st.session_state["pending_mfa_user"] = user
                return True

            # Create session
            session_token = self.create_session_token(user)
            user.session_token = session_token

            st.session_state["authenticated"] = True
            st.session_state["user"] = user

            return True

        except Exception as e:
            logger.exception(f"Facebook OAuth error: {e}")
            return False

    # Multi-Factor Authentication
    def setup_mfa(self, user: User) -> tuple[str, str]:
        """Setup MFA for user.

        Args:
            user: User to setup MFA for

        Returns:
            (secret, qr_code_url)
        """
        try:
            import pyotp

            # Generate secret
            secret = pyotp.random_base32()

            # Create provisioning URI
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=user.email, issuer_name=self.config.mfa_issuer_name
            )

            return secret, provisioning_uri

        except ImportError:
            logger.error("pyotp not installed. Install with: pip install pyotp qrcode")
            return "", ""

    def verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify MFA code.

        Args:
            secret: User's MFA secret
            code: Code to verify

        Returns:
            True if valid
        """
        try:
            import pyotp

            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)

        except ImportError:
            return False

    def render_mfa_setup(self, user: User) -> None:
        """Render MFA setup UI.

        Args:
            user: User to setup MFA for
        """
        if not self.config.enable_mfa:
            return

        st.subheader("Multi-Factor Authentication Setup")

        if "mfa_secret" not in st.session_state:
            secret, provisioning_uri = self.setup_mfa(user)
            st.session_state["mfa_secret"] = secret
            st.session_state["mfa_uri"] = provisioning_uri

        secret = st.session_state["mfa_secret"]
        provisioning_uri = st.session_state["mfa_uri"]

        if provisioning_uri:
            try:
                import qrcode
                from io import BytesIO

                # Generate QR code
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(provisioning_uri)
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")
                buf = BytesIO()
                img.save(buf, format="PNG")

                st.image(buf.getvalue())
                st.code(secret, language=None)
                st.info("Scan the QR code or enter the secret in your authenticator app")

            except ImportError:
                st.code(provisioning_uri)
                st.warning("Install qrcode for QR code display: pip install qrcode")

        # Verify code
        mfa_code = st.text_input("Enter verification code:", max_chars=6)
        if st.button("Verify"):
            if self.verify_mfa_code(secret, mfa_code):
                user.mfa_verified = True
                st.session_state["user"] = user
                st.session_state["authenticated"] = True

                # Save MFA secret (in production, encrypt and store in database)
                st.session_state["user_mfa_secret"] = secret

                st.success("MFA setup complete!")
                st.rerun()
            else:
                st.error("Invalid code. Please try again.")
