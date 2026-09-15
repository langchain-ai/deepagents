"""Tests for `dcode://` link validation, registration, and handling."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

from deepagents_code.url_scheme import (
    MAX_PROMPT_CHARS,
    URL_SCHEME,
    RegistrationError,
    TerminalChoice,
    UrlRequestError,
    _linux,
    _macos,
    _windows,
    build_open_url,
    handler,
    handler_status,
    parse_open_url,
    resolve_launcher,
)
from deepagents_code.url_scheme.registration import install_handler, uninstall_handler

THREAD_ID = str(uuid.uuid4())


@pytest.fixture
def project(tmp_path: Path) -> Path:
    directory = tmp_path / "project"
    directory.mkdir()
    return directory.resolve()


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home


class TestParseAccepts:
    def test_directory_only(self, project: Path) -> None:
        assert parse_open_url(f"dcode://open?dir={project}").directory == project

    def test_opaque_form_without_authority(self, project: Path) -> None:
        """`dcode:open?...` is what some link authors write; both spellings work."""
        request = parse_open_url(f"dcode:open?dir={project}")
        assert request.directory == project

    def test_trailing_slash_after_action(self, project: Path) -> None:
        assert parse_open_url(f"dcode://open/?dir={project}").directory == project

    def test_scheme_is_case_insensitive(self, project: Path) -> None:
        assert parse_open_url(f"DCode://open?dir={project}").directory == project

    def test_all_parameters(self, project: Path) -> None:
        url = build_open_url(
            project, thread=THREAD_ID, agent="research", prompt="review the diff"
        )
        request = parse_open_url(url)
        assert request.directory == project
        assert request.thread == THREAD_ID
        assert request.agent == "research"
        assert request.prompt == "review the diff"

    def test_plus_in_a_path_needs_percent_encoding(self, tmp_path: Path) -> None:
        """Form decoding turns `+` into a space, as a browser would."""
        directory = tmp_path / "c++proj"
        directory.mkdir()
        encoded = str(directory).replace("+", "%2B")
        assert parse_open_url(f"dcode://open?dir={encoded}").directory == (
            directory.resolve()
        )

    def test_percent_and_plus_encoding_decode(self, project: Path) -> None:
        request = parse_open_url(f"dcode://open?dir={project}&prompt=a+b%2Bc%20d")
        assert request.prompt == "a b+c d"

    def test_tilde_expands_to_home(self, fake_home: Path) -> None:
        (fake_home / "proj").mkdir()
        request = parse_open_url("dcode://open?dir=~/proj")
        assert request.directory == (fake_home / "proj").resolve()

    def test_symlinked_directory_resolves_to_target(self, tmp_path: Path) -> None:
        """The confirmation has to name the directory that is actually opened."""
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        assert parse_open_url(f"dcode://open?dir={link}").directory == target.resolve()

    def test_blank_prompt_is_dropped(self, project: Path) -> None:
        assert parse_open_url(f"dcode://open?dir={project}&prompt=%20").prompt is None

    def test_multiline_prompt_is_kept(self, project: Path) -> None:
        url = build_open_url(project, prompt="first\nsecond")
        assert parse_open_url(url).prompt == "first\nsecond"

    def test_thread_id_is_canonicalized(self, project: Path) -> None:
        upper = THREAD_ID.upper()
        assert parse_open_url(f"dcode://open?dir={project}&thread={upper}").thread == (
            THREAD_ID
        )

    def test_round_trips_through_build(self, project: Path) -> None:
        request = parse_open_url(
            build_open_url(project, thread=THREAD_ID, agent="a b", prompt="x&y=z?")
        )
        assert request.agent == "a b"
        assert request.prompt == "x&y=z?"
        assert request.thread == THREAD_ID


class TestParseRefuses:
    def test_other_scheme(self) -> None:
        with pytest.raises(UrlRequestError, match="Not a dcode:// link"):
            parse_open_url("https://example.com")

    def test_unknown_action(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="Unsupported action 'run'"):
            parse_open_url(f"dcode://run?dir={project}")

    def test_missing_action(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="names no action"):
            parse_open_url(f"dcode://?dir={project}")

    def test_authority_syntax_in_action(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="not a recognized action name"):
            parse_open_url(f"dcode://user@open?dir={project}")

    def test_path_after_action(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="takes no path"):
            parse_open_url(f"dcode://open/extra?dir={project}")

    def test_fragment(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="do not take a fragment"):
            parse_open_url(f"dcode://open?dir={project}#frag")

    def test_unknown_parameter(self, project: Path) -> None:
        """Forward compatibility is traded away on purpose: unknown keys fail."""
        with pytest.raises(UrlRequestError, match="Unsupported parameter 'yolo'"):
            parse_open_url(f"dcode://open?dir={project}&yolo=1")

    @pytest.mark.parametrize(
        "smuggled",
        ["auto-approve=1", "approval=yolo", "sandbox=daytona", "model=gpt-5.5"],
    )
    def test_session_altering_parameters(self, project: Path, smuggled: str) -> None:
        with pytest.raises(UrlRequestError, match="Unsupported parameter"):
            parse_open_url(f"dcode://open?dir={project}&{smuggled}")

    def test_repeated_parameter(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="appears more than once"):
            parse_open_url(f"dcode://open?dir={project}&dir={project}")

    def test_missing_dir(self) -> None:
        with pytest.raises(UrlRequestError, match="missing the 'dir' parameter"):
            parse_open_url("dcode://open?agent=research")

    def test_relative_dir(self) -> None:
        with pytest.raises(UrlRequestError, match="must be an absolute path"):
            parse_open_url("dcode://open?dir=../elsewhere")

    def test_missing_dir_on_disk(self, tmp_path: Path) -> None:
        with pytest.raises(UrlRequestError, match="does not exist or is unreadable"):
            parse_open_url(f"dcode://open?dir={tmp_path / 'absent'}")

    def test_unencoded_plus_explains_itself(self, tmp_path: Path) -> None:
        """The decoded path gains a space, which is a confusing symptom alone."""
        directory = tmp_path / "c++proj"
        directory.mkdir()
        with pytest.raises(UrlRequestError, match=r"encoded as %2B"):
            parse_open_url(f"dcode://open?dir={directory}")

    def test_dir_that_is_a_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(UrlRequestError, match="is not a directory"):
            parse_open_url(f"dcode://open?dir={target}")

    def test_non_uuid_thread(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="not a valid thread id"):
            parse_open_url(f"dcode://open?dir={project}&thread=latest")

    def test_resume_sentinel_is_unreachable(self, project: Path) -> None:
        """`-r __MOST_RECENT__` is not a thread a link's author can name."""
        with pytest.raises(UrlRequestError, match="not a valid thread id"):
            parse_open_url(f"dcode://open?dir={project}&thread=__MOST_RECENT__")

    def test_agent_with_path_separator(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="not a usable agent name"):
            parse_open_url(f"dcode://open?dir={project}&agent=../../etc")

    @pytest.mark.parametrize("reserved", ["bin", "Bin", "plugins"])
    def test_reserved_agent_name(self, project: Path, reserved: str) -> None:
        with pytest.raises(UrlRequestError, match="reserved"):
            parse_open_url(f"dcode://open?dir={project}&agent={reserved}")

    @pytest.mark.parametrize("control", ["%1B", "%00", "%07", "%C2%9B"])
    def test_control_characters_in_prompt(self, project: Path, control: str) -> None:
        """An escape sequence could redraw the confirmation it is shown in."""
        with pytest.raises(UrlRequestError, match="control character"):
            parse_open_url(f"dcode://open?dir={project}&prompt=hi{control}there")

    def test_control_characters_in_dir(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="control character"):
            parse_open_url(f"dcode://open?dir={project}%00/etc")

    def test_bidi_override_in_prompt(self, project: Path) -> None:
        """Displayed text and submitted text must not be able to disagree."""
        url = f"dcode://open?dir={project}&prompt=delete%20%E2%80%AEtxt.evil"
        with pytest.raises(UrlRequestError, match="hidden or direction-changing"):
            parse_open_url(url)

    def test_overlong_prompt(self, project: Path) -> None:
        long_prompt = "x" * (MAX_PROMPT_CHARS + 1)
        with pytest.raises(UrlRequestError, match="prompt is too long"):
            parse_open_url(build_open_url(project, prompt=long_prompt))

    def test_overlong_url(self, project: Path) -> None:
        with pytest.raises(UrlRequestError, match="Link is too long"):
            parse_open_url(f"dcode://open?dir={project}&prompt={'x' * 9000}")


class TestResolveLauncher:
    def test_prefers_absolute_argv0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = tmp_path / "dcode"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [str(script), "url", "install"])
        assert resolve_launcher() == script

    def test_falls_back_to_path_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`python -m deepagents_code` reports a module file, not a command."""
        script = tmp_path / "dcode"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        monkeypatch.setenv("PATH", str(tmp_path))
        assert resolve_launcher() == script

    def test_raises_when_nothing_is_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["__main__.py"])
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(RegistrationError, match="Could not find the dcode command"):
            resolve_launcher()


class TestUnsupportedPlatform:
    def test_status_reports_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "sunos5")
        status = handler_status()
        assert not status.supported
        assert not status.installed
        assert "sunos5" in status.detail

    def test_install_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "sunos5")
        with pytest.raises(RegistrationError, match="sunos5"):
            install_handler()

    def test_uninstall_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "sunos5")
        with pytest.raises(RegistrationError, match="sunos5"):
            uninstall_handler()


class TestLinuxBackend:
    @pytest.mark.usefixtures("fake_home")
    def test_desktop_entry_declares_the_scheme(self) -> None:
        entry = _linux._desktop_entry(Path("/usr/bin/dcode"))
        assert f"MimeType=x-scheme-handler/{URL_SCHEME};" in entry
        assert 'Exec="/usr/bin/dcode" url open %u' in entry
        # A TUI needs a terminal window, which the desktop supplies.
        assert "Terminal=true" in entry

    def test_exec_quotes_a_path_with_spaces(self) -> None:
        assert (
            _linux._exec_value(Path("/opt/my tools/dcode"))
            == '"/opt/my tools/dcode" url open %u'
        )

    @pytest.mark.parametrize("bad", ['/opt/we"ird/dcode', "/opt/back\\slash/dcode"])
    def test_exec_refuses_unrepresentable_paths(self, bad: str) -> None:
        with pytest.raises(RegistrationError, match="Exec line cannot carry"):
            _linux._exec_value(Path(bad))

    def test_install_writes_entry_and_mimeapps_fallback(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without `xdg-utils`, the association is written directly."""
        monkeypatch.setattr(_linux.shutil, "which", lambda _name: None)
        entry = _linux.install(Path("/usr/bin/dcode"))

        assert (
            entry == fake_home / ".local/share/applications/dcode-url-handler.desktop"
        )
        assert entry.is_file()
        mimeapps = (fake_home / ".config/mimeapps.list").read_text(encoding="utf-8")
        assert "[Default Applications]" in mimeapps
        assert f"x-scheme-handler/{URL_SCHEME}={_linux.DESKTOP_FILE_NAME}" in mimeapps

    @pytest.mark.usefixtures("fake_home")
    def test_install_honors_xdg_data_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_linux.shutil, "which", lambda _name: None)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
        entry = _linux.install(Path("/usr/bin/dcode"))
        assert entry.parent == tmp_path / "xdg-data" / "applications"

    def test_install_uses_xdg_mime_when_present(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(_linux.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(
            _linux,
            "_run",
            lambda argv, *, what: calls.append(argv) or True,  # noqa: ARG005
        )
        _linux.install(Path("/usr/bin/dcode"))

        assert [
            "/usr/bin/xdg-mime",
            "default",
            _linux.DESKTOP_FILE_NAME,
            f"x-scheme-handler/{URL_SCHEME}",
        ] in calls
        # `xdg-mime` owns the association, so mimeapps.list is left alone.
        assert not (fake_home / ".config/mimeapps.list").exists()

    def test_mimeapps_edit_preserves_other_entries(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_linux.shutil, "which", lambda _name: None)
        mimeapps = fake_home / ".config" / "mimeapps.list"
        mimeapps.parent.mkdir(parents=True)
        mimeapps.write_text(
            "[Default Applications]\ntext/html=firefox.desktop\n", encoding="utf-8"
        )

        _linux.install(Path("/usr/bin/dcode"))
        assert "text/html=firefox.desktop" in mimeapps.read_text(encoding="utf-8")

        _linux.uninstall()
        after = mimeapps.read_text(encoding="utf-8")
        assert "text/html=firefox.desktop" in after
        assert _linux.DESKTOP_FILE_NAME not in after

    def test_uninstall_leaves_a_foreign_association_alone(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_linux.shutil, "which", lambda _name: None)
        mimeapps = fake_home / ".config" / "mimeapps.list"
        mimeapps.parent.mkdir(parents=True)
        mimeapps.write_text(
            f"[Default Applications]\nx-scheme-handler/{URL_SCHEME}=other.desktop\n",
            encoding="utf-8",
        )
        _linux.uninstall()
        assert "other.desktop" in mimeapps.read_text(encoding="utf-8")

    @pytest.mark.usefixtures("fake_home")
    def test_uninstall_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_linux.shutil, "which", lambda _name: None)
        assert _linux.uninstall() == []
        _linux.install(Path("/usr/bin/dcode"))
        assert len(_linux.uninstall()) >= 1
        assert _linux.uninstall() == []

    def test_status_notices_a_foreign_default(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_linux.shutil, "which", lambda _name: None)
        _linux.install(Path("/usr/bin/dcode"))
        mimeapps = fake_home / ".config" / "mimeapps.list"
        mimeapps.write_text(
            f"[Default Applications]\nx-scheme-handler/{URL_SCHEME}=other.desktop\n",
            encoding="utf-8",
        )
        status = _linux.status()
        assert status.installed
        assert status.default_handler == "other.desktop"
        assert "other.desktop" in status.detail


class TestMacosBackend:
    def test_applet_quotes_the_launcher_path(self) -> None:
        source = _macos._applet_source(
            Path("/opt/my tools/dcode"), TerminalChoice.TERMINAL
        )
        assert "'/opt/my tools/dcode' url open" in source
        # The link itself is quoted by AppleScript at dispatch time.
        assert "quoted form of this_URL" in source

    def test_applet_escapes_quotes_in_the_launcher_path(self) -> None:
        """A quote in the path must survive both quoting layers intact."""
        source = _macos._applet_source(Path("/opt/we'ird/dcode"), TerminalChoice.ITERM)
        # `shlex.quote` closes and reopens the single-quoted run around the
        # quote; each resulting double quote is then escaped for AppleScript.
        assert r"""'/opt/we'\"'\"'ird/dcode'""" in source
        assert "iTerm" in source

    def test_applet_source_is_ascii(self) -> None:
        """`osacompile` reads the generated source; ASCII keeps that portable."""
        source = _macos._applet_source(Path("/usr/bin/dcode"), TerminalChoice.TERMINAL)
        assert source.isascii()

    def test_explicit_iterm_without_iterm_reports_why(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_macos, "_iterm_available", lambda: False)
        with pytest.raises(RegistrationError, match="iTerm does not appear"):
            _macos._resolve_terminal(TerminalChoice.ITERM)

    def test_auto_falls_back_when_iterm_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code._env_vars import LAUNCH_TERM_PROGRAM

        monkeypatch.setenv(LAUNCH_TERM_PROGRAM, "iTerm.app")
        monkeypatch.setattr(_macos, "_iterm_available", lambda: False)
        assert _macos._resolve_terminal(TerminalChoice.AUTO) is TerminalChoice.TERMINAL

    def test_auto_picks_iterm_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepagents_code._env_vars import LAUNCH_TERM_PROGRAM

        monkeypatch.setenv(LAUNCH_TERM_PROGRAM, "iTerm.app")
        monkeypatch.setattr(_macos, "_iterm_available", lambda: True)
        assert _macos._resolve_terminal(TerminalChoice.AUTO) is TerminalChoice.ITERM

    def test_metadata_declares_the_scheme(self, tmp_path: Path) -> None:
        import plistlib

        bundle = tmp_path / "dcode.app"
        plist_path = bundle / "Contents" / "Info.plist"
        plist_path.parent.mkdir(parents=True)
        with plist_path.open("wb") as handle:
            plistlib.dump({"CFBundleName": "applet"}, handle)

        _macos._write_bundle_metadata(
            bundle, launcher=Path("/usr/bin/dcode"), terminal=TerminalChoice.TERMINAL
        )
        plist = plistlib.loads(plist_path.read_bytes())
        assert plist["CFBundleURLTypes"][0]["CFBundleURLSchemes"] == [URL_SCHEME]
        assert plist["CFBundleIdentifier"] == _macos.BUNDLE_ID
        # Browsers label their prompt with the display name.
        assert plist["CFBundleDisplayName"] == "dcode"

    def test_refuses_to_replace_another_application(self, tmp_path: Path) -> None:
        import plistlib

        bundle = tmp_path / "dcode.app"
        plist_path = bundle / "Contents" / "Info.plist"
        plist_path.parent.mkdir(parents=True)
        with plist_path.open("wb") as handle:
            plistlib.dump({"CFBundleIdentifier": "com.example.other"}, handle)

        with pytest.raises(RegistrationError, match="belongs to another application"):
            _macos._clear_existing_bundle(bundle)

    @pytest.mark.usefixtures("fake_home")
    def test_uninstall_refuses_a_foreign_bundle(self) -> None:
        import plistlib

        bundle = _macos.app_path()
        plist_path = bundle / "Contents" / "Info.plist"
        plist_path.parent.mkdir(parents=True)
        with plist_path.open("wb") as handle:
            plistlib.dump({"CFBundleIdentifier": "com.example.other"}, handle)

        with pytest.raises(RegistrationError, match="not dcode's URL handler"):
            _macos.uninstall()
        assert bundle.exists()

    @pytest.mark.usefixtures("fake_home")
    def test_macos_uninstall_is_idempotent(self) -> None:
        assert _macos.uninstall() == ()

    @pytest.mark.skipif(sys.platform != "darwin", reason="needs osacompile")
    @pytest.mark.usefixtures("fake_home")
    def test_generated_applet_compiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Guards the AppleScript itself: a syntax slip only shows up here."""
        if not _macos._OSACOMPILE.is_file():
            pytest.skip("osacompile is unavailable")
        # Keep Launch Services out of the test's way; the applet build is the
        # subject, and registering a temp-directory bundle is a real side effect.
        monkeypatch.setattr(_macos, "_LSREGISTER", Path("/nonexistent/lsregister"))

        bundle = _macos.install(
            Path("/usr/bin/dcode"), terminal=TerminalChoice.TERMINAL
        )
        assert bundle.is_dir()
        status = _macos.status()
        assert status.installed
        assert status.handler_path == str(bundle)
        assert _macos.uninstall() == (str(bundle),)


class TestWindowsBackend:
    def test_key_path_is_per_user(self) -> None:
        assert f"Software\\Classes\\{URL_SCHEME}" == _windows.KEY_PATH


class TestHandler:
    def test_refuses_a_bad_link_without_launching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(
            handler, "_launch", lambda _request: pytest.fail("must not launch")
        )
        assert handler.open_from_url("dcode://open?dir=relative") == (
            handler.EXIT_REFUSED
        )

    def test_declines_when_the_terminal_cannot_be_asked(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail closed: no interactive terminal means no approval, so no launch."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(
            handler, "_launch", lambda _request: pytest.fail("must not launch")
        )
        assert handler.open_from_url(f"dcode://open?dir={project}") == (
            handler.EXIT_DECLINED
        )

    def test_declines_when_the_user_says_no(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(handler, "_confirm", lambda _request: False)
        monkeypatch.setattr(
            handler, "_launch", lambda _request: pytest.fail("must not launch")
        )
        assert handler.open_from_url(f"dcode://open?dir={project}") == (
            handler.EXIT_DECLINED
        )

    def test_launches_in_the_requested_directory(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launcher = tmp_path / "dcode"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        recorded: dict[str, object] = {}

        def fake_execv(path: str, argv: list[str]) -> None:
            recorded["path"] = path
            recorded["argv"] = argv
            raise SystemExit(0)

        monkeypatch.setattr(handler, "_confirm", lambda _request: True)
        monkeypatch.setattr(handler, "resolve_launcher", lambda: launcher)
        monkeypatch.setattr(
            os, "chdir", lambda target: recorded.setdefault("cwd", target)
        )
        monkeypatch.setattr(os, "execv", fake_execv)

        url = build_open_url(project, thread=THREAD_ID, agent="research", prompt="hi")
        with pytest.raises(SystemExit):
            handler.open_from_url(url)

        assert recorded["cwd"] == project
        assert recorded["argv"] == [
            str(launcher),
            "-a",
            "research",
            "-r",
            THREAD_ID,
            "-m",
            "hi",
        ]

    def test_launch_argv_carries_no_approval_flags(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A link must not be able to widen what the session may do unattended."""
        launcher = tmp_path / "dcode"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        recorded: list[str] = []

        def fake_execv(_path: str, argv: list[str]) -> None:
            recorded.extend(argv)
            raise SystemExit(0)

        monkeypatch.setattr(handler, "_confirm", lambda _request: True)
        monkeypatch.setattr(handler, "resolve_launcher", lambda: launcher)
        monkeypatch.setattr(os, "chdir", lambda _target: None)
        monkeypatch.setattr(os, "execv", fake_execv)

        with pytest.raises(SystemExit):
            handler.open_from_url(build_open_url(project, prompt="go"))

        forbidden = {"-y", "--auto-approve", "--yolo", "--sandbox", "-M", "--model"}
        assert forbidden.isdisjoint(recorded)

    def test_reports_a_missing_dcode_command(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_missing() -> Path:
            msg = "Could not find the dcode command to register."
            raise RegistrationError(msg)

        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(handler, "_confirm", lambda _request: True)
        monkeypatch.setattr(handler, "resolve_launcher", raise_missing)
        assert handler.open_from_url(f"dcode://open?dir={project}") == (
            handler.EXIT_REFUSED
        )

    def test_typed_confirmation_requires_an_explicit_yes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda *_args: "y")
        assert not handler._confirm_by_typing("Open this session")
        monkeypatch.setattr("builtins.input", lambda *_args: " YES ")
        assert handler._confirm_by_typing("Open this session")

    def test_typed_confirmation_declines_on_eof(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_eof(*_args: object) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert not handler._confirm_by_typing("Open this session")

    def test_prompt_preview_summarizes_a_long_tail(self) -> None:
        lines = handler._prompt_preview("\n".join(f"line {n}" for n in range(30)))
        assert len(lines) == handler._PROMPT_PREVIEW_LINES + 1
        assert "more line" in lines[-1]
