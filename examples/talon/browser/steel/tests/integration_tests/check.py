"""Exercise a disposable synthetic-only Docker profile without publishing ports."""

import json
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "talon-steel-check-" + uuid.uuid4().hex[:10]
IMAGE = "talon-steel-foundation"
VOLUME = NAME + "-profile"
CAPS = [
    item
    for cap in ("CHOWN", "SETUID", "SETGID", "SETPCAP")
    for item in ("--cap-add", cap)
]


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args], check=False, capture_output=True, text=True
    )
    if check and result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def start(name: str, *, nonroot: bool = False) -> None:
    identity = ["--user", "1000:1000"] if nonroot else CAPS
    gate = "false" if nonroot else "true"
    docker(
        "run",
        "-d",
        "--name",
        name,
        "--network",
        "none",
        "--env",
        f"STEEL_REQUIRE_EGRESS_READY={gate}",
        "--cap-drop",
        "ALL",
        *identity,
        "--security-opt",
        "no-new-privileges",
        "--mount",
        f"type=volume,source={VOLUME},target=/var/lib/steel/profile",
        IMAGE,
    )


def node(source: str) -> str:
    return docker("exec", NAME, "node", "--input-type=module", "-e", source).stdout


def ready() -> None:
    for _ in range(60):
        result = docker(
            "exec",
            NAME,
            "node",
            "-e",
            "fetch('http://127.0.0.1:9222/json/version').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))",
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise AssertionError(docker("logs", NAME).stdout)


def verify_browser() -> None:
    source = r"""
import fs from 'node:fs';
const browsers = fs.readdirSync('/proc').filter(n=>/^\d+$/.test(n)).flatMap(n=>{
  try { const args=fs.readFileSync(`/proc/${n}/cmdline`,'utf8').split('\0');
    return args[0]==='/usr/lib/chromium/chromium' && !args.some(a=>a.startsWith('--type=')) ? [args] : [];
  } catch { return []; }
});
console.log(JSON.stringify(browsers));
"""
    browsers = json.loads(node(source))
    assert len(browsers) == 1, docker("top", NAME, "-eo", "pid,args").stdout
    args = browsers[0]
    for expected in (
        "--user-data-dir=/var/lib/steel/profile",
        "--proxy-server=http://172.30.14.3:8080",
        "--proxy-bypass-list=<-loopback>",
        "--disable-quic",
        "--disable-extensions",
        "--remote-debugging-port=9222",
    ):
        assert expected in args, args
    assert not any(
        arg.startswith(
            (
                "--load-extension",
                "--disable-extensions-except",
                "--unsafely-treat-insecure-origin-as-secure",
            )
        )
        for arg in args
    ), args
    node("""
import p from 'puppeteer-core';
import http from 'node:http';
const b=await p.connect({browserURL:'http://127.0.0.1:9222'});
let hits=0;
const server=http.createServer((req,res)=>{hits++; res.end('bypass');});
await new Promise(resolve=>server.listen(18080,'0.0.0.0',resolve));
try {
  for (const url of ['http://localhost:18080/', 'http://127.0.0.1:18080/', 'http://127.0.0.1:3000/', 'http://127.0.0.1:9222/json/version', 'http://10.0.0.1/', 'http://169.254.169.254/latest/meta-data/']) {
    const page=await b.newPage();
    let rejected=false;
    try { await page.goto(url,{timeout:10000}); }
    catch(e) { if(!/ERR_PROXY_CONNECTION_FAILED|ERR_BLOCKED_BY_ADMINISTRATOR/.test(e.message))throw e; rejected=true; }
    if(!rejected)throw Error('navigation bypassed proxy: '+url);
    await page.close();
  }
  if(hits)throw Error('localhost received browser traffic');
} finally { await b.disconnect(); await new Promise(resolve=>server.close(resolve)); }
""")


def main() -> None:
    try:
        docker("build", "-t", IMAGE, str(ROOT))
        start(NAME)
        time.sleep(2)
        assert "node" not in docker("top", NAME, "-eo", "pid,args").stdout
        docker(
            "exec",
            NAME,
            "python3",
            "-c",
            "from pathlib import Path; p=Path('/run/steel-network'); p.mkdir(); (p/'ready').touch()",
        )
        ready()
        verify_browser()
        metadata = docker(
            "exec", NAME, "stat", "-c", "%u:%a", "/var/lib/steel/profile"
        ).stdout.strip()
        assert metadata == "1000:700", metadata
        connection = "import p from 'puppeteer-core'; const b=await p.connect({browserURL:'http://127.0.0.1:9222'});"
        node(
            connection
            + "await (await b.pages())[0].setCookie({name:'synthetic',value:'persisted',domain:'example.test',path:'/',expires:Math.floor(Date.now()/1000)+3600}); await b.disconnect();"
        )
        start(NAME + "-locked")
        assert docker("wait", NAME + "-locked").stdout.strip() == "1"
        docker("stop", "-t", "35", NAME)
        assert (
            docker("inspect", "-f", "{{.State.ExitCode}}", NAME).stdout.strip() == "0"
        )
        docker("rm", NAME)
        start(NAME, nonroot=True)
        ready()
        verify_browser()
        node(
            connection
            + "if(!(await (await b.pages())[0].cookies('http://example.test')).some(c=>c.name==='synthetic'&&c.value==='persisted'))throw Error('cookie missing'); await b.disconnect();"
        )
        node(
            "const r=await fetch('http://127.0.0.1:3000/v1/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dimensions:{width:1280,height:720},userAgent:'TalonSynthetic/1.0',timezone:'UTC'})}); if(!r.ok)throw Error(await r.text());"
        )
        verify_browser()
        node(
            connection
            + "if(await (await b.pages())[0].evaluate(()=>navigator.userAgent)!=='TalonSynthetic/1.0')throw Error('user agent ignored'); await b.disconnect();"
        )
        node(
            "const r=await fetch('http://127.0.0.1:3000/v1/sessions/release',{method:'POST'}); if(!r.ok)throw Error(await r.text());"
        )
        time.sleep(2)
        processes = docker("top", NAME, "-eo", "pid,args").stdout
        assert "/usr/lib/chromium/chromium" not in processes, processes
        docker("stop", "-t", "35", NAME)
        docker(
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            "--mount",
            f"type=volume,source={VOLUME},target=/var/lib/steel/profile",
            IMAGE,
            "-c",
            "from pathlib import Path; Path('/var/lib/steel/profile/Local State').write_text('corrupt')",
        )
        docker("start", NAME)
        assert docker("wait", NAME).stdout.strip() == "1"
        print(
            "PASS profile path/owner/mode, cookie restart, exclusive lock, release without relaunch, corrupt rejection"
        )
    finally:
        docker("rm", "-f", NAME, NAME + "-locked", check=False)
        docker("volume", "rm", VOLUME, check=False)


if __name__ == "__main__":
    main()
