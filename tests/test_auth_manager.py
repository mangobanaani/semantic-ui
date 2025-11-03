"""Tests for Authentication Manager."""

import os
from datetime import datetime
from unittest.mock import patch

import pytest

from semantic_kernel_ui.auth.auth_manager import AuthConfig, AuthManager, User


@pytest.fixture
def clean_env():
    """Clean environment for testing."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def jwt_env():
    """Environment with JWT secret."""
    with patch.dict(os.environ, {"JWT_SECRET": "test-secret-key"}, clear=True):
        yield


@pytest.fixture
def oauth_env():
    """Environment with OAuth settings."""
    with patch.dict(os.environ, {
        "ENABLE_GOOGLE_OAUTH": "true",
        "GOOGLE_CLIENT_ID": "test-client-id",
        "GOOGLE_REDIRECT_URI": "http://localhost:8501",
        "ENABLE_FACEBOOK_OAUTH": "true",
        "FACEBOOK_APP_ID": "test-app-id",
        "FACEBOOK_REDIRECT_URI": "http://localhost:8501",
    }, clear=True):
        yield


class TestAuthConfig:
    """Test AuthConfig functionality."""

    def test_default_config(self, clean_env):
        """Test default configuration values."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.enable_auth is False
        assert config.jwt_secret == ""
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration_hours == 24


class TestUser:
    """Test User model."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            email="test@example.com",
            name="Test User",
            provider="jwt",
            authenticated_at=datetime.utcnow(),
        )

        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.provider == "jwt"
        assert user.picture is None

    def test_user_with_picture(self):
        """Test creating a user with picture."""
        user = User(
            email="test@example.com",
            name="Test User",
            picture="https://example.com/pic.jpg",
            provider="google",
            authenticated_at=datetime.utcnow(),
        )

        assert user.picture == "https://example.com/pic.jpg"


class TestAuthManager:
    """Test AuthManager functionality."""

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_initialization(self, mock_session):
        """Test auth manager initialization."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        assert manager.config == config
        assert "authenticated" in mock_session
        assert "user" in mock_session

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_is_authenticated_disabled(self, mock_session):
        """Test authentication check when disabled."""
        config = AuthConfig(_env_file=None, enable_auth=False)  # type: ignore[call-arg]
        manager = AuthManager(config)

        # Should always return True when auth is disabled
        assert manager.is_authenticated() is True

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    @patch.dict(os.environ, {"ENABLE_AUTH": "true"}, clear=True)
    def test_is_authenticated_enabled_not_logged_in(self, mock_session):
        """Test authentication check when enabled but not logged in."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        assert manager.is_authenticated() is False

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_is_authenticated_enabled_logged_in(self, mock_session):
        """Test authentication check when enabled and logged in."""
        config = AuthConfig(_env_file=None, enable_auth=True)  # type: ignore[call-arg]
        manager = AuthManager(config)

        mock_session["authenticated"] = True
        assert manager.is_authenticated() is True

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_create_jwt_token(self, mock_session, jwt_env):
        """Test JWT token creation."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        token = manager.create_jwt_token("test@example.com", "Test User")

        assert isinstance(token, str)
        assert len(token) > 0

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_create_jwt_token_no_secret(self, mock_session):
        """Test JWT token creation without secret."""
        config = AuthConfig(_env_file=None, jwt_secret="")  # type: ignore[call-arg]
        manager = AuthManager(config)

        with pytest.raises(ValueError, match="JWT_SECRET is not configured"):
            manager.create_jwt_token("test@example.com", "Test User")

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_verify_jwt_token_valid(self, mock_session, jwt_env):
        """Test JWT token verification with valid token."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        token = manager.create_jwt_token("test@example.com", "Test User")
        payload = manager.verify_jwt_token(token)

        assert payload is not None
        assert payload["email"] == "test@example.com"
        assert payload["name"] == "Test User"

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_verify_jwt_token_invalid(self, mock_session, jwt_env):
        """Test JWT token verification with invalid token."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        payload = manager.verify_jwt_token("invalid-token")

        assert payload is None

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_verify_jwt_token_no_secret(self, mock_session, clean_env):
        """Test JWT token verification without secret."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        payload = manager.verify_jwt_token("some-token")

        assert payload is None

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_authenticate_with_jwt_success(self, mock_session, jwt_env):
        """Test JWT authentication success."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        token = manager.create_jwt_token("test@example.com", "Test User")
        success, error = manager.authenticate_with_jwt(token)

        assert success is True
        assert mock_session["authenticated"] is True
        assert mock_session["user"].email == "test@example.com"

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_authenticate_with_jwt_invalid_token(self, mock_session, jwt_env):
        """Test JWT authentication with invalid token."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        success, error = manager.authenticate_with_jwt("invalid-token")

        assert success is False
        assert error == "Authentication failed"

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-key", "ALLOWED_USERS": "test@example.com"}, clear=True)
    def test_authenticate_with_jwt_restricted_user_allowed(self, mock_session):
        """Test JWT authentication with allowed user."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        token = manager.create_jwt_token("test@example.com", "Test User")
        success, error = manager.authenticate_with_jwt(token)

        assert success is True

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    @patch.dict(os.environ, {"JWT_SECRET": "test-secret-key", "ALLOWED_USERS": "allowed@example.com"}, clear=True)
    def test_authenticate_with_jwt_restricted_user_denied(self, mock_session):
        """Test JWT authentication with denied user."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        token = manager.create_jwt_token("denied@example.com", "Denied User")
        success, error = manager.authenticate_with_jwt(token)

        assert success is False

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_logout(self, mock_session, jwt_env):
        """Test logout functionality."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        # First login
        token = manager.create_jwt_token("test@example.com", "Test User")
        manager.authenticate_with_jwt(token)

        assert mock_session["authenticated"] is True

        # Then logout
        manager.logout()

        assert mock_session["authenticated"] is False
        assert mock_session["user"] is None

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_get_current_user(self, mock_session, jwt_env):
        """Test getting current user."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        # No user initially
        assert manager.get_current_user() is None

        # Login and check user
        token = manager.create_jwt_token("test@example.com", "Test User")
        manager.authenticate_with_jwt(token)

        user = manager.get_current_user()
        assert user is not None
        assert user.email == "test@example.com"

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_is_user_allowed_no_restrictions(self, mock_session):
        """Test user allowed check with no restrictions."""
        config = AuthConfig(_env_file=None, allowed_users="")  # type: ignore[call-arg]
        manager = AuthManager(config)

        assert manager._is_user_allowed("anyone@example.com") is True

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    @patch.dict(os.environ, {"ALLOWED_USERS": "user1@example.com, user2@example.com"}, clear=True)
    def test_is_user_allowed_with_restrictions(self, mock_session):
        """Test user allowed check with restrictions."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        assert manager._is_user_allowed("user1@example.com") is True
        assert manager._is_user_allowed("user2@example.com") is True
        assert manager._is_user_allowed("user3@example.com") is False

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_is_user_allowed_case_insensitive(self, mock_session):
        """Test user allowed check is case insensitive."""
        config = AuthConfig(_env_file=None, allowed_users="User@Example.com")  # type: ignore[call-arg]
        manager = AuthManager(config)

        assert manager._is_user_allowed("user@example.com") is True
        assert manager._is_user_allowed("USER@EXAMPLE.COM") is True

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_get_google_oauth_url(self, mock_session, oauth_env):
        """Test Google OAuth URL generation."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        url = manager.get_google_oauth_url()

        assert "accounts.google.com" in url
        assert "test-client-id" in url
        assert "localhost" in url

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_get_google_oauth_url_disabled(self, mock_session):
        """Test Google OAuth URL when disabled."""
        config = AuthConfig(_env_file=None, enable_google_oauth=False)  # type: ignore[call-arg]
        manager = AuthManager(config)

        url = manager.get_google_oauth_url()

        assert url == ""

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_get_facebook_oauth_url(self, mock_session, oauth_env):
        """Test Facebook OAuth URL generation."""
        config = AuthConfig(_env_file=None)  # type: ignore[call-arg]
        manager = AuthManager(config)

        url = manager.get_facebook_oauth_url()

        assert "facebook.com" in url
        assert "test-app-id" in url
        assert "localhost" in url

    @patch("semantic_kernel_ui.auth.auth_manager.st.session_state", new_callable=dict)
    def test_get_facebook_oauth_url_disabled(self, mock_session):
        """Test Facebook OAuth URL when disabled."""
        config = AuthConfig(_env_file=None, enable_facebook_oauth=False)  # type: ignore[call-arg]
        manager = AuthManager(config)

        url = manager.get_facebook_oauth_url()

        assert url == ""
