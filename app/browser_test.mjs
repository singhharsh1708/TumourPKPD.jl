// End-to-end check in a real (headless) Chrome against a running explorer: drives the widgets through the
// DevTools protocol and measures how long the status line takes to change after an input event.
//   node app/browser_test.mjs http://127.0.0.1:8081 [screenshot.png]
// IDLE=0 skips waiting for the virtual trial between actions (worst case: trial and main plot compete for threads).
// Needs Node >= 22 (global WebSocket) and Google Chrome. Start the server first (run.jl).
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const url = process.argv[2] ?? 'http://127.0.0.1:8081';
const shot = process.argv[3];
const chrome = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500);
const proc = spawn(chrome, ['--headless=new', `--remote-debugging-port=${port}`, '--use-gl=angle', '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--window-size=1400,1600', '--user-data-dir=/tmp/tp-chrome-' + port, 'about:blank'],
  { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.length) break; } catch {}
  await sleep(200);
}
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send('Page.enable');
await send('Page.navigate', { url });
// wait for the status line and the two WebGL canvases
let ready = false;
for (let i = 0; i < 150 && !ready; i++) {
  await sleep(400);
  ready = await ev(`document.querySelectorAll('canvas').length >= 2`);
}
console.log('canvases ready:', ready);
const status = `(() => { const n = [...document.querySelectorAll('.tp-note')].find(e => e.textContent.startsWith('main plot updated')); return n ? n.textContent : null })()`;
const banner = `(() => { const b = document.querySelector('.tp-main div div'); return b ? b.textContent.slice(0, 90) : null })()`;
console.log('initial status:', await ev(status));
const trialIdle = async () => {
  for (let i = 0; i < 600; i++) {
    const t = await ev(`(() => { const n = [...document.querySelectorAll('.tp-note')].find(e => /Virtual PDX trial|computing|trial hidden/.test(e.textContent)); return n ? n.textContent : '' })()`);
    if (/Virtual PDX trial/.test(t) && (t.includes('Updated in') || t.includes('Computed at'))) return;
    await sleep(200);
  }
};
// force an update, then time input -> status change
async function timed(label, js) {
  const before = await ev(status);
  const t0 = performance.now();
  await ev(js);
  let after = before;
  for (let i = 0; i < 400 && after === before; i++) { await sleep(10); after = await ev(status); }
  const dt = performance.now() - t0;
  if (process.env.IDLE !== '0') { await sleep(1200); await trialIdle(); }
  console.log(label.padEnd(34), (dt).toFixed(0).padStart(5), 'ms |', after, '| banner:', await ev(banner));
  return dt;
}
const setSlider = (k, idx) => `(() => { const s = document.querySelectorAll('input[type=range]')[${k}]; s.valueAsNumber = ${idx}; s.dispatchEvent(new Event('input', {bubbles: true})); })()`;
const setSelect = (k, idx) => `(() => { const s = document.querySelectorAll('select')[${k}]; s.selectedIndex = ${idx}; s.dispatchEvent(new Event('change', {bubbles: true})); })()`;
const times = [];
times.push(await timed('dose multiplier -> 2.0x', setSlider(0, 36)));
times.push(await timed('dose multiplier -> 0.5x', setSlider(0, 6)));
times.push(await timed('schedule -> q2d', setSelect(2, 2)));
times.push(await timed('schedule -> 5-on-2-off', setSelect(2, 3)));
times.push(await timed('duration -> 21 d', setSlider(1, 21)));
times.push(await timed('horizon -> 90 d', setSlider(2, 77)));
times.push(await timed('PDX -> New PDX (population)', setSelect(1, 0)));
times.push(await timed('drug -> BKM120', setSelect(0, 1)));
times.push(await timed('drug -> BYL719', setSelect(0, 2)));
console.log('round trip median ms:', times.sort((a, b) => a - b)[Math.floor(times.length / 2)].toFixed(0));
await sleep(6000);
if (shot) {
  const r = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  writeFileSync(shot, Buffer.from(r.result.data, 'base64'));
  console.log('screenshot ->', shot);
}
ws.close(); proc.kill();
process.exit(0);
