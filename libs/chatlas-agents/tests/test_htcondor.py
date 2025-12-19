"""Tests for HTCondor integration."""

import shutil
import pytest
from pathlib import Path
from chatlas_agents.htcondor import HTCondorJobSubmitter


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory for tests."""
    return tmp_path / "htcondor_jobs"


def test_htcondor_submitter_initialization(temp_output_dir):
    """Test HTCondor submitter initialization."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    assert submitter.docker_image == "python:3.13-slim"
    assert submitter.output_dir == temp_output_dir
    assert submitter.output_dir.exists()


def test_generate_submit_file_basic(temp_output_dir):
    """Test basic submit file generation."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-job"
    prompt = "Hello, world!"
    
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt=prompt,
    )
    
    assert submit_file.exists()
    assert submit_file.name == f"{job_name}.sub"
    
    # Read and verify content
    content = submit_file.read_text()
    assert "HTCondor submit file" in content
    assert job_name in content
    assert "python -m chatlas_agents.cli run" in content
    assert "--input" in content
    # Verify prompt is present (may be shell-quoted)
    assert "Hello, world!" in content or "'Hello, world!'" in content
    assert "--docker-sandbox" in content
    assert "universe = docker" in content
    assert "python:3.13-slim" in content


def test_generate_submit_file_with_config(temp_output_dir):
    """Test submit file generation with config file."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-job-config"
    prompt = "Test prompt"
    config_file = "/path/to/config.yaml"
    
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt=prompt,
        config_file=config_file,
    )
    
    content = submit_file.read_text()
    assert "--config /path/to/config.yaml" in content


def test_generate_submit_file_with_env_vars(temp_output_dir):
    """Test submit file generation with environment variables."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-job-env"
    prompt = "Test prompt"
    env_vars = {
        "CHATLAS_LLM_API_KEY": "test-key",
        "CHATLAS_LLM_PROVIDER": "openai",
    }
    
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt=prompt,
        env_vars=env_vars,
    )
    
    content = submit_file.read_text()
    assert "environment = " in content
    assert "CHATLAS_LLM_API_KEY=test-key" in content
    assert "CHATLAS_LLM_PROVIDER=openai" in content


def test_generate_submit_file_with_special_chars_in_env(temp_output_dir):
    """Test submit file generation with special characters in environment variables."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-job-env-special"
    prompt = "Test prompt"
    env_vars = {
        "API_KEY": "test$key",
        "MESSAGE": 'test"value',
    }
    
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt=prompt,
        env_vars=env_vars,
    )
    
    content = submit_file.read_text()
    assert "environment = " in content
    # $ and " should be escaped with backslashes
    assert "API_KEY=test\\$key" in content
    assert 'MESSAGE=test\\"value' in content


def test_generate_submit_file_with_custom_resources(temp_output_dir):
    """Test submit file generation with custom resource requirements."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-job-resources"
    prompt = "Test prompt"
    
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt=prompt,
        request_cpus=4,
        request_memory="8GB",
        request_disk="10GB",
    )
    
    content = submit_file.read_text()
    assert "request_cpus = 4" in content
    assert "request_memory = 8GB" in content
    assert "request_disk = 10GB" in content


def test_generate_submit_file_paths(temp_output_dir):
    """Test that submit file contains correct paths for logs and output."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-job-paths"
    prompt = "Test prompt"
    
    submit_file = submitter.generate_submit_file(
        job_name=job_name,
        prompt=prompt,
    )
    
    content = submit_file.read_text()
    job_dir = submitter.output_dir / job_name
    
    assert f"output = {job_dir}/job.$(ClusterId).$(ProcId).out" in content
    assert f"error = {job_dir}/job.$(ClusterId).$(ProcId).err" in content
    assert f"log = {job_dir}/job.$(ClusterId).log" in content
    assert "queue 1" in content


@pytest.mark.skipif(
    not shutil.which('condor_submit'),
    reason="HTCondor not installed"
)
def test_submit_job_dry_run(temp_output_dir):
    """Test job submission in dry run mode (no actual submission)."""
    submitter = HTCondorJobSubmitter(
        docker_image="python:3.13-slim",
        output_dir=temp_output_dir,
    )
    
    job_name = "test-dry-run"
    prompt = "Test prompt"
    
    # Dry run should not raise errors
    cluster_id = submitter.submit_job(
        job_name=job_name,
        prompt=prompt,
        dry_run=True,
    )
    
    # Dry run should return None
    assert cluster_id is None
    
    # But submit file should exist
    submit_file = submitter.output_dir / job_name / f"{job_name}.sub"
    assert submit_file.exists()
