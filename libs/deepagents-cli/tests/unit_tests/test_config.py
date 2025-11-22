"""Tests for config module including project discovery utilities."""

from pathlib import Path

from deepagents_cli.config import _find_agents_md, _find_project_agent_md, _find_project_root


class TestProjectRootDetection:
    """Test project root detection via .git directory."""

    def test_find_project_root_with_git(self, tmp_path: Path) -> None:
        """Test that project root is found when .git directory exists."""
        # Create a mock project structure
        project_root = tmp_path / "my-project"
        project_root.mkdir()
        git_dir = project_root / ".git"
        git_dir.mkdir()

        # Create a subdirectory to search from
        subdir = project_root / "src" / "components"
        subdir.mkdir(parents=True)

        # Should find project root from subdirectory
        result = _find_project_root(subdir)
        assert result == project_root

    def test_find_project_root_no_git(self, tmp_path: Path) -> None:
        """Test that None is returned when no .git directory exists."""
        # Create directory without .git
        no_git_dir = tmp_path / "no-git"
        no_git_dir.mkdir()

        result = _find_project_root(no_git_dir)
        assert result is None

    def test_find_project_root_nested_git(self, tmp_path: Path) -> None:
        """Test that nearest .git directory is found (not parent repos)."""
        # Create nested git repos
        outer_repo = tmp_path / "outer"
        outer_repo.mkdir()
        (outer_repo / ".git").mkdir()

        inner_repo = outer_repo / "inner"
        inner_repo.mkdir()
        (inner_repo / ".git").mkdir()

        # Should find inner repo, not outer
        result = _find_project_root(inner_repo)
        assert result == inner_repo


class TestProjectAgentMdFinding:
    """Test finding project-specific agent.md files."""

    def test_find_agent_md_in_deepagents_dir(self, tmp_path: Path) -> None:
        """Test finding agent.md in .deepagents/ directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create .deepagents/agent.md
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        agent_md = deepagents_dir / "agent.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_find_agent_md_in_root(self, tmp_path: Path) -> None:
        """Test finding agent.md in project root (fallback)."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create root-level agent.md (no .deepagents/)
        agent_md = project_root / "agent.md"
        agent_md.write_text("Project instructions")

        result = _find_project_agent_md(project_root)
        assert len(result) == 1
        assert result[0] == agent_md

    def test_both_agent_md_files_combined(self, tmp_path: Path) -> None:
        """Test that both agent.md files are returned when both exist."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Create both locations
        deepagents_dir = project_root / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "agent.md"
        deepagents_md.write_text("In .deepagents/")

        root_md = project_root / "agent.md"
        root_md.write_text("In root")

        # Should return both, with .deepagents/ first
        result = _find_project_agent_md(project_root)
        assert len(result) == 2
        assert result[0] == deepagents_md
        assert result[1] == root_md

    def test_find_agent_md_not_found(self, tmp_path: Path) -> None:
        """Test that empty list is returned when no agent.md exists."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        result = _find_project_agent_md(project_root)
        assert result == []


class TestAgentsMdFinding:
    """Test finding AGENTS.md files hierarchically."""

    def test_find_agents_md_in_current_dir(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md in current directory."""
        current_dir = tmp_path / "current"
        current_dir.mkdir()

        # Create AGENTS.md in current directory
        agents_md = current_dir / "AGENTS.md"
        agents_md.write_text("Agent instructions")

        result = _find_agents_md(start_path=current_dir)
        assert len(result) == 1
        assert result[0] == agents_md

    def test_find_agents_md_hierarchical(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md files hierarchically from nested directories."""
        # Create directory structure: project/subdir/current
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        subdir = project_root / "subdir"
        subdir.mkdir()

        current_dir = subdir / "current"
        current_dir.mkdir()

        # Create AGENTS.md in multiple levels
        current_agents_md = current_dir / "AGENTS.md"
        current_agents_md.write_text("Current dir instructions")

        subdir_agents_md = subdir / "AGENTS.md"
        subdir_agents_md.write_text("Subdir instructions")

        # Search from current_dir, should find both but not reach project root
        result = _find_agents_md(start_path=current_dir, project_root=project_root)
        assert len(result) == 2
        assert result[0] == current_agents_md  # Most specific first
        assert result[1] == subdir_agents_md

    def test_find_agents_md_stops_at_project_root(self, tmp_path: Path) -> None:
        """Test that search stops before reaching project root."""
        # Create directory structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        subdir = project_root / "subdir"
        subdir.mkdir()

        # Create AGENTS.md in both project root and subdir
        project_agents_md = project_root / "AGENTS.md"
        project_agents_md.write_text("Project root instructions")

        subdir_agents_md = subdir / "AGENTS.md"
        subdir_agents_md.write_text("Subdir instructions")

        # Search from subdir, should NOT include project root
        result = _find_agents_md(start_path=subdir, project_root=project_root)
        assert len(result) == 1
        assert result[0] == subdir_agents_md
        assert project_agents_md not in result

    def test_find_agents_md_no_project_root(self, tmp_path: Path) -> None:
        """Test finding AGENTS.md without a project root constraint."""
        # Create nested directories without .git
        parent = tmp_path / "parent"
        parent.mkdir()

        child = parent / "child"
        child.mkdir()

        # Create AGENTS.md in both
        parent_agents_md = parent / "AGENTS.md"
        parent_agents_md.write_text("Parent instructions")

        child_agents_md = child / "AGENTS.md"
        child_agents_md.write_text("Child instructions")

        # Search without project root should find both
        result = _find_agents_md(start_path=child, project_root=None)
        assert len(result) == 2
        assert result[0] == child_agents_md
        assert result[1] == parent_agents_md

    def test_find_agents_md_not_found(self, tmp_path: Path) -> None:
        """Test that empty list is returned when no AGENTS.md exists."""
        current_dir = tmp_path / "current"
        current_dir.mkdir()

        result = _find_agents_md(start_path=current_dir)
        assert result == []

    def test_find_agents_md_monorepo_scenario(self, tmp_path: Path) -> None:
        """Test AGENTS.md in monorepo with nested projects."""
        # Create monorepo structure
        monorepo = tmp_path / "monorepo"
        monorepo.mkdir()
        (monorepo / ".git").mkdir()

        # Create package structure
        packages = monorepo / "packages"
        packages.mkdir()

        pkg_a = packages / "pkg-a"
        pkg_a.mkdir()

        pkg_a_src = pkg_a / "src"
        pkg_a_src.mkdir()

        # Create AGENTS.md at different levels
        packages_agents_md = packages / "AGENTS.md"
        packages_agents_md.write_text("Packages general instructions")

        pkg_a_agents_md = pkg_a / "AGENTS.md"
        pkg_a_agents_md.write_text("Package A specific instructions")

        pkg_a_src_agents_md = pkg_a_src / "AGENTS.md"
        pkg_a_src_agents_md.write_text("Package A source instructions")

        # Search from pkg-a/src, should get all three (from src up to packages)
        result = _find_agents_md(start_path=pkg_a_src, project_root=monorepo)
        assert len(result) == 3
        assert result[0] == pkg_a_src_agents_md
        assert result[1] == pkg_a_agents_md
        assert result[2] == packages_agents_md
