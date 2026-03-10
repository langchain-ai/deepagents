from deepagents_codex.status import CodexAuthInfo, CodexAuthStatus


class TestCodexAuthStatus:
    def test_enum_values(self) -> None:
        assert CodexAuthStatus.PACKAGE_MISSING.value == "package_missing"
        assert CodexAuthStatus.NOT_AUTHENTICATED.value == "not_authenticated"
        assert CodexAuthStatus.AUTHENTICATED.value == "authenticated"
        assert CodexAuthStatus.EXPIRED.value == "expired"
        assert CodexAuthStatus.REFRESH_FAILED.value == "refresh_failed"
        assert CodexAuthStatus.CORRUPT.value == "corrupt"


class TestCodexAuthInfo:
    def test_is_usable_authenticated(self) -> None:
        info = CodexAuthInfo(status=CodexAuthStatus.AUTHENTICATED)
        assert info.is_usable is True

    def test_is_usable_expired(self) -> None:
        info = CodexAuthInfo(status=CodexAuthStatus.EXPIRED)
        assert info.is_usable is True

    def test_is_usable_not_authenticated(self) -> None:
        info = CodexAuthInfo(status=CodexAuthStatus.NOT_AUTHENTICATED)
        assert info.is_usable is False

    def test_is_usable_package_missing(self) -> None:
        info = CodexAuthInfo(status=CodexAuthStatus.PACKAGE_MISSING)
        assert info.is_usable is False

    def test_is_usable_refresh_failed(self) -> None:
        info = CodexAuthInfo(status=CodexAuthStatus.REFRESH_FAILED)
        assert info.is_usable is False

    def test_is_usable_corrupt(self) -> None:
        info = CodexAuthInfo(status=CodexAuthStatus.CORRUPT)
        assert info.is_usable is False

    def test_fields(self) -> None:
        info = CodexAuthInfo(
            status=CodexAuthStatus.AUTHENTICATED,
            user_email="user@example.com",
            expires_at=1700000000.0,
            message="All good",
        )
        assert info.user_email == "user@example.com"
        assert info.expires_at == 1700000000.0
        assert info.message == "All good"
