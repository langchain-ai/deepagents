import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';

const UPSTREAM = 'http://172.30.14.2:3000';
const CAST = '/v1/sessions/cast';
const COOKIE = 'talon_local';
const INPUT_LIMIT = 16 * 1024;
const FRAME_LIMIT = 2 * 1024 * 1024;
const pageId = (value) => typeof value === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(value);
const record = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const keys = (value, required, optional = []) => record(value) && required.every((key) => Object.hasOwn(value, key)) && Object.keys(value).every((key) => [...required, ...optional].includes(key));
const number = (value, min, max) => Number.isFinite(value) && value >= min && value <= max;
const integer = (value, min, max) => Number.isInteger(value) && number(value, min, max);
const text = (value, max) => typeof value === 'string' && value.length <= max;

function validInput(value) {
  if (!keys(value, ['type', 'pageId', 'event']) || !pageId(value.pageId)) return false;
  const event = value.event;
  if (value.type === 'mouseEvent') {
    return keys(event, ['type', 'x', 'y', 'button', 'modifiers'], ['clickCount', 'deltaX', 'deltaY']) &&
      ['mousePressed', 'mouseReleased', 'mouseMoved', 'mouseWheel'].includes(event.type) &&
      number(event.x, 0, 16384) && number(event.y, 0, 16384) && ['none', 'left', 'middle', 'right'].includes(event.button) &&
      integer(event.modifiers, 0, 15) && (!Object.hasOwn(event, 'clickCount') || integer(event.clickCount, 0, 3)) &&
      ['deltaX', 'deltaY'].every((key) => !Object.hasOwn(event, key) || number(event[key], -16384, 16384));
  }
  return value.type === 'keyEvent' && keys(event, ['type', 'code', 'key', 'keyCode'], ['text', 'modifiers']) &&
    ['keyDown', 'keyUp', 'char'].includes(event.type) && text(event.code, 64) && text(event.key, 64) && integer(event.keyCode, 0, 65535) &&
    (!Object.hasOwn(event, 'text') || text(event.text, 64)) && (!Object.hasOwn(event, 'modifiers') || integer(event.modifiers, 0, 15));
}

async function readBody(request) {
  let body = '';
  for await (const chunk of request) {
    body += chunk.toString();
    if (Buffer.byteLength(body) > INPUT_LIMIT) throw new Error('body_limit');
  }
  return body;
}

async function defaultFetchHTML() {
  const response = await fetch(`${UPSTREAM}/v1/sessions/debug?interactive=true&pageIndex=0&showControls=false`, { redirect: 'error', signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error('viewer_unavailable');
  const chunks = [];
  let size = 0;
  for await (const chunk of response.body) {
    size += chunk.length;
    if (size > FRAME_LIMIT) throw new Error('html_limit');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

export function createLocalViewer({ coordinator, WebSocket, WebSocketServer, origin, token, fetchHTML = defaultFetchHTML, createInput = null }) {
  if (origin !== 'http://127.0.0.1:8765' || typeof token !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(token)) throw new Error('invalid_local_viewer_config');
  const wsOrigin = origin.replace('http:', 'ws:');
  const wss = new WebSocketServer({ noServer: true, maxPayload: INPUT_LIMIT, perMessageDeflate: false });
  const sessions = new Map();
  const pairs = new Set();
  let local = null;
  let busy = false;
  let closed = false;
  let closePromise;
  let loginAttempts = 0;
  let loginWindow = [];

  function send(response, status, body, type = 'text/plain; charset=utf-8', extra = {}) {
    response.writeHead(status, {
      'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'same-origin', 'X-Frame-Options': 'SAMEORIGIN',
      'Content-Security-Policy': `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src ${wsOrigin} ${origin}; frame-src 'self'; frame-ancestors 'self'; form-action 'self'; base-uri 'none'; object-src 'none'`,
      ...extra,
    });
    response.end(body);
  }

  function session(request) {
    const cookies = (request.headers.cookie || '').split(';').map((part) => part.trim()).filter((part) => part.startsWith(`${COOKIE}=`));
    if (cookies.length !== 1) return null;
    const id = cookies[0].slice(COOKIE.length + 1);
    const value = sessions.get(id);
    if (!value) return null;
    if (Date.now() >= value.created + 1800000 || Date.now() >= value.touched + 600000) {
      void revoke(value).catch(() => {});
      return null;
    }
    value.touched = Date.now();
    return value;
  }

  function controlling(value) {
    coordinator.expire();
    return value && !value.revoked && Date.now() < value.created + 1800000 && Date.now() < value.touched + 600000 && local?.session === value && coordinator.lease === local.lease && local.lease.mode === 'HUMAN' && !local.lease.expiring && !local.draining;
  }

  function closeSocket(socket) {
    if (socket.readyState === 3) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { reject(new Error('close_timeout')); socket.terminate(); }, 3000);
      socket.once('close', () => { clearTimeout(timer); resolve(); });
      try { socket.close(1000); } catch { socket.terminate(); }
    });
  }

  function deadline(operation) {
    let timer;
    return Promise.race([
      new Promise((resolve) => resolve(operation())),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('input_timeout')), 10000); }),
    ]).finally(() => clearTimeout(timer));
  }

  function fail(current) {
    coordinator.failed(current.lease);
    try { current.input?.abort(); } catch { }
  }

  function drain(current = local) {
    if (!current) return Promise.resolve();
    if (!current.draining) {
      if (current.lease.mode !== 'FAILED') coordinator.fence(current.lease, 'PAUSED');
      const sockets = [...pairs].filter((pair) => pair.local === current);
      current.draining = Promise.resolve().then(async () => {
        await Promise.allSettled([...current.pending]);
        try { await deadline(() => current.input?.close()); }
        finally { await Promise.all(sockets.flatMap((pair) => [closeSocket(pair.client), closeSocket(pair.upstream)])); }
      });
      current.draining.catch(() => fail(current));
    }
    return current.draining;
  }

  async function fence(failed = false, current = local) {
    if (!current) return;
    if (failed) fail(current);
    else if (current.lease.mode !== 'FAILED') coordinator.fence(current.lease, 'PAUSED');
    try { await drain(current); } catch { }
  }

  async function revoke(value, current = local) {
    value.revoked = true;
    for (const [id, session] of sessions) if (session === value) sessions.delete(id);
    if (!current || current.session !== value) return;
    if (!current.revoking) current.revoking = (async () => {
      try {
        await current.setup?.catch(() => {});
        await drain(current);
        await coordinator.stop(current.lease, true);
        if (local === current) local = null;
      } catch (error) {
        fail(current);
        throw error;
      }
    })();
    return current.revoking;
  }

  function inject(current, value, pair, message) {
    if (!current.input || current.pending.size >= 32) { void fence(true, current); return; }
    const operation = current.queue.then(async () => {
      if (!controlling(value) || local !== current || pair.pageId !== message.pageId) return;
      if (current.target !== message.pageId) {
        const result = await deadline(() => current.input.command('Target.attachToTarget', { targetId: message.pageId, flatten: true }));
        if (!result || typeof result.sessionId !== 'string' || !result.sessionId) throw new Error('invalid_input_session');
        current.target = message.pageId;
        current.sessionId = result.sessionId;
      }
      if (!controlling(value) || local !== current || pair.pageId !== message.pageId) return;
      const event = message.event;
      const params = message.type === 'mouseEvent' ? {
        type: event.type, x: event.x, y: event.y, button: event.button,
        buttons: event.button === 'none' ? 0 : 1, clickCount: event.clickCount || 1,
        modifiers: event.modifiers || 0, deltaX: event.deltaX, deltaY: event.deltaY,
      } : {
        type: event.type, text: event.text, unmodifiedText: event.text ? event.text.toLowerCase() : undefined,
        code: event.code, key: event.key, windowsVirtualKeyCode: event.keyCode, nativeVirtualKeyCode: event.keyCode,
        modifiers: event.modifiers || 0, autoRepeat: false, isKeypad: false, isSystemKey: false,
      };
      await deadline(() => current.input.command(message.type === 'mouseEvent' ? 'Input.dispatchMouseEvent' : 'Input.dispatchKeyEvent', params, current.sessionId));
    });
    current.pending.add(operation);
    current.queue = operation.catch(() => { void fence(true, current); }).finally(() => current.pending.delete(operation));
  }

  const sweep = setInterval(() => {
    for (const [id, value] of sessions) {
      if (Date.now() >= value.created + 1800000 || Date.now() >= value.touched + 600000) {
        void revoke(value).catch(() => {});
      }
    }
    if (local && (coordinator.lease !== local.lease || local.lease.mode !== 'HUMAN' || local.lease.expiring)) void fence(local.lease.mode === 'FAILED');
  }, 1000);
  sweep.unref();

  async function take(value) {
    if (local || coordinator.status().mode !== 'IDLE' || coordinator.failedLatch) throw new Error('unavailable');
    const [provider, sender_id] = Object.entries(coordinator.config.identities)[0];
    const owner = { operator_id: coordinator.config.operator, provider, sender_id, conversation_id: `local:${randomUUID()}`, run_id: `local:${randomUUID()}`, background: false };
    const acquired = await coordinator.action({ action: 'acquire', request_id: randomUUID(), owner });
    local = { session: value, lease: coordinator.lease, draining: null, input: null, pending: new Set(), queue: Promise.resolve() };
    const current = local;
    current.setup = (async () => {
      if (value.revoked) throw new Error('session_revoked');
      const handoff = await coordinator.action({ ...acquired, owner, action: 'handoff', request_id: randomUUID() });
      if (value.revoked) throw new Error('session_revoked');
      await coordinator.take({ ...handoff, owner }, owner);
      if (value.revoked) throw new Error('session_revoked');
      current.input = createInput ? createInput(current.lease, () => local === current && controlling(value)) : null;
    })();
    try { await current.setup; }
    catch (error) {
      if (value.revoked) await revoke(value, current);
      else await fence(true, current);
      throw error;
    }
  }

  function shell(value) {
    const linkLogin = `<script>
async function loginFromLink(){
if(!location.hash)return;
const fragment=new URLSearchParams(location.hash.slice(1));
history.replaceState(null,'','/');
if(!document.querySelector('form'))return;
const token=fragment.get('token')||'', message=document.querySelector('#login-message');
if([...fragment.keys()].length!==1||!/^[A-Za-z0-9_-]{43}$/.test(token)){message.textContent='Invalid or expired sign-in link.';return;}
try{const response=await fetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({token})});
if(response.ok&&response.redirected){location.replace('/');return;}}catch{}
message.textContent='Invalid or expired sign-in link.';
}
addEventListener('hashchange',loginFromLink);loginFromLink();
</script>`;
    if (!value) return '<!doctype html><title>Local browser login</title><h1>Local browser</h1><form method="post" action="/auth/login"><label>Launch password <input type="password" name="token" required autocomplete="off" maxlength="43"></label><button>Sign in</button></form><p id="login-message" role="status"></p>' + linkLogin;
    return `<!doctype html><title>Local browser control</title><h1>Local browser</h1><button id="take">Take</button> <button id="release">Release</button> <button id="logout">Sign out</button><p>Local exclusive control only; Take requires an idle browser.</p><p id="status" role="status"></p><iframe title="Steel browser viewer" hidden style="width:100%;height:80vh;border:0" sandbox="allow-scripts allow-same-origin"></iframe>${linkLogin}<script>
const frame=document.querySelector('iframe'), status=document.querySelector('#status');
async function refresh(){const response=await fetch('/state');if(response.status===401){location.reload();return;}const state=await response.json();document.querySelector('#take').disabled=!state.available;document.querySelector('#release').disabled=!state.owned;status.textContent=state.controlling?'You control this browser':state.owned?'Paused: release to finish':state.available?'Available':'Unavailable';if(state.controlling){if(!frame.hasAttribute('src'))frame.src='/viewer';frame.hidden=false;}else{frame.removeAttribute('src');frame.hidden=true;}}
for(const action of ['take','release','logout'])document.querySelector('#'+action).onclick=async()=>{const response=await fetch(action==='logout'?'/auth/logout':'/'+action,{method:'POST',headers:{'X-CSRF-Token':'${value.csrf}'}});if(action==='logout'&&response.ok){location.reload();return;}if(!response.ok)status.textContent='Request failed; browser control is unavailable.';await refresh();};
refresh();setInterval(refresh,2000);
</script>`;
  }

  async function handler(request, response) {
    try {
      if (closed || request.headers.host !== '127.0.0.1:8765') return send(response, 403, 'Forbidden');
      const route = request.url;
      if (!['/', '/auth/login', '/auth/logout', '/state', '/take', '/release', '/viewer'].includes(route)) return send(response, 404, 'Not found');
      const mutation = ['/auth/login', '/auth/logout', '/take', '/release'].includes(route);
      if (request.method !== (mutation ? 'POST' : 'GET')) return send(response, 405, 'Method not allowed');
      if (mutation && request.headers.origin !== origin) return send(response, 403, 'Forbidden');
      if (route === '/auth/login') {
          const now = Date.now();
          loginWindow = loginWindow.filter((time) => now - time < 60000);
          if (loginAttempts >= 50 || loginWindow.length >= 5) return send(response, 429, 'Login limit');
          loginAttempts += 1;
          loginWindow.push(now);
        if (request.headers['content-type'] !== 'application/x-www-form-urlencoded') return send(response, 400, 'Invalid login');
        const form = new URLSearchParams(await readBody(request));
        const supplied = form.get('token') || '';
        if ([...form.keys()].length !== 1 || !/^[A-Za-z0-9_-]{43}$/.test(supplied) || !timingSafeEqual(Buffer.from(supplied), Buffer.from(token))) return send(response, 403, 'Forbidden');
        if (sessions.size >= 64) return send(response, 429, 'Session limit');
        const id = randomBytes(32).toString('base64url');
        sessions.set(id, { created: Date.now(), touched: Date.now(), csrf: randomBytes(32).toString('base64url') });
        return send(response, 303, '', 'text/plain', { Location: '/', 'Set-Cookie': `${COOKIE}=${id}; HttpOnly; SameSite=Strict; Path=/; Max-Age=1800` });
      }
      const value = session(request);
      if (route === '/') return send(response, 200, shell(value), 'text/html; charset=utf-8');
      if (!value) return send(response, 401, 'Sign in required');
      if (route === '/state') return send(response, 200, JSON.stringify({ available: !local && coordinator.status().mode === 'IDLE' && !coordinator.failedLatch && !busy, owned: local?.session === value, controlling: Boolean(controlling(value)) }), 'application/json');
      if (route === '/viewer') {
        if (!controlling(value)) return send(response, 403, 'Forbidden');
        const html = await fetchHTML();
        const stock = "const baseWsUrl = 'ws://0.0.0.0:3000/v1/sessions/cast?pageIndex=0';";
        if (typeof html !== 'string' || Buffer.byteLength(html) > FRAME_LIMIT || html.split(stock).length !== 2 || !html.includes('const singlePageMode = true;') || html.includes('id="url-text"') || html.includes('id="tab-bar"')) throw new Error('unexpected_viewer_template');
        if (!controlling(value)) return send(response, 403, 'Forbidden');
        return send(response, 200, html.replace(stock, `const baseWsUrl = '${wsOrigin}${CAST}?pageIndex=0';`), 'text/html; charset=utf-8');
      }
      if (request.headers['x-csrf-token'] !== value.csrf) return send(response, 403, 'Forbidden');
      if ((await readBody(request)) !== '') return send(response, 400, 'Unexpected body');
      if (route === '/auth/logout') {
          await revoke(value);
          return send(response, 200, '{}', 'application/json', { 'Set-Cookie': `${COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0` });
        }
        if (busy) return send(response, 409, 'Busy');
      busy = true;
      try {
        if (route === '/take') await take(value);
        else {
          if (!local || local.session !== value) return send(response, 403, 'Forbidden');
          coordinator.fence(local.lease, local.lease.mode === 'FAILED' ? 'FAILED' : 'PAUSED');
          await drain();
          await coordinator.stop(local.lease, true);
          local = null;
        }
        return send(response, 200, '{}', 'application/json');
      } finally { busy = false; }
    } catch { if (!response.headersSent) send(response, 409, 'Local viewer unavailable'); else response.destroy(); }
  }

  function upgrade(request, socket, head) {
    const reject = () => { socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n'); };
    try {
      if (closed || request.method !== 'GET' || request.headers.host !== '127.0.0.1:8765' || request.headers.origin !== origin || request.headers['sec-websocket-protocol']) return reject();
      const value = session(request);
      if (!controlling(value)) return reject();
      const match = /^\/v1\/sessions\/cast\?(tabInfo=true|pageIndex=0|pageId=([A-Za-z0-9_-]{1,128}))$/.exec(request.url);
      if (!match) return reject();
      const discovery = match[1] === 'tabInfo=true';
      if ([...pairs].some((pair) => pair.discovery === discovery) || pairs.size >= 2) return reject();
      wss.handleUpgrade(request, socket, head, (client) => {
        if (!controlling(value)) { client.close(); return; }
        let upstream;
        try { upstream = new WebSocket(`${UPSTREAM.replace('http:', 'ws:')}${CAST}?${match[1]}`, { maxPayload: FRAME_LIMIT, perMessageDeflate: false, followRedirects: false, handshakeTimeout: 5000 }); }
        catch { client.close(); void fence(true); return; }
        const current = local;
        const pair = { client, upstream, discovery, local: current, pageId: null };
        pairs.add(pair);
        let credits = 240;
        let last = Date.now();
        let frameAt = 0;
        let clientClosed = false;
        let upstreamClosed = false;
        const stop = () => { void fence(false, current); };
        client.on('error', stop);
        upstream.on('error', stop);
        client.on('close', () => { clientClosed = true; if (upstreamClosed) pairs.delete(pair); stop(); });
        upstream.on('close', () => { upstreamClosed = true; if (clientClosed) pairs.delete(pair); stop(); });
        client.on('message', (data, binary) => {
          try {
            if (!controlling(value)) return stop();
            const now = Date.now();
            credits = Math.min(240, credits + (now - last) * 0.12);
            last = now;
            if (binary || discovery || data.length > INPUT_LIMIT || credits < 1 || upstream.readyState !== 1 || upstream.bufferedAmount > INPUT_LIMIT) return stop();
            const message = JSON.parse(data.toString());
            if (!validInput(message)) return stop();
            if (match[1] === 'pageIndex=0' && message.pageId === 'default') message.pageId = pair.pageId;
            if (!pair.pageId || message.pageId !== pair.pageId || (match[2] && message.pageId !== match[2])) return stop();
            credits -= 1;
            value.touched = now;
            inject(current, value, pair, message);
          } catch { stop(); }
        });
        upstream.on('message', (data, binary) => {
          try {
            if (!controlling(value)) return stop();
            if (binary || data.length > FRAME_LIMIT || client.bufferedAmount > FRAME_LIMIT) return stop();
            const message = JSON.parse(data.toString());
            if (!record(message)) return stop();
            if (Object.hasOwn(message, 'data')) {
              if (discovery || !pageId(message.pageId) || message.pageId === 'default' || (match[2] && message.pageId !== match[2])) return stop();
              pair.pageId = message.pageId;
              if (Date.now() - frameAt < 1000 / 15) return;
              frameAt = Date.now();
            }
            if (client.readyState === 1) client.send(data, { binary: false }, (error) => { if (error) stop(); });
          } catch { stop(); }
        });
      });
    } catch { reject(); }
  }

  function close() {
    if (!closePromise) closePromise = (async () => {
      closed = true;
      clearInterval(sweep);
      sessions.clear();
      await fence(true);
      await new Promise((resolve) => wss.close(resolve));
    })();
    return closePromise;
  }

  return { handler, upgrade, close };
}
