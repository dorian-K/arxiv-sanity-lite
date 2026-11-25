// the service worker (served as /sw.js, registered by app.js): the site opens without a network,
// on a plane, and quickly on a slow connection. data always comes from the server (lib.js shows
// a kept copy meanwhile on a slow one, marked as such); only the app itself comes from the device.
//   pages (any address of the app): the server's html, a copy kept; offline, or when the server
//     takes over SLOW_PAGE ms, that copy (the html is the same for every address)
//   versioned static files (?v=, vendor/): from the device once fetched (they never change
//     under their url); versions the kept html doesn't name are dropped when it is replaced
//   the reading pack (static/app/offline.js writes it into PACK): PDFs from the device first,
//     thumbnails from the server, else the device. light copies (X-Resanity-Packed) are stored
//     packed by static/app/shrink.js and unpacked here (unpack: LZMA, then the Type 1 fonts'
//     eexec cipher redone), so the reader gets the PDF itself
// a broken service worker could keep the site from loading: keep this small. to take it off
// everyone's browsers, serve one whose install calls self.registration.unregister()
const SHELL = 'shell-v1';
const STATIC = 'static-v1';
const PACK = 'pack-v1'; // as in offline.js
const PACKED = 'X-Resanity-Packed'; // as in offline.js
try {
    importScripts('/static/vendor/lzma-2.3.2/src/lzma-d-min.js'); // self.LZMA
} catch (e) {
    // light copies can't be opened, everything else works
}
const SLOW_PAGE = 3000; // ms the server gets for a page before the copy on the device is shown
const APP_PATHS = ['/', '/digest', '/profile', '/import', '/inspect', '/stats', '/agents', '/history', '/graph', '/offline', '/read', '/about', '/privacy', '/imprint'];

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', e => {
    const req = e.request;
    const url = new URL(req.url);
    if (req.method !== 'GET' || url.origin !== self.location.origin) return;
    if (req.mode === 'navigate' && APP_PATHS.includes(url.pathname)) e.respondWith(page(req, e));
    else if (url.pathname.startsWith('/static/') && (url.searchParams.has('v') || url.pathname.startsWith('/static/vendor/'))) e.respondWith(staticFile(req, url, e));
    else if (url.pathname.startsWith('/offline/pdf/')) e.respondWith(fromPack(req, true));
    else if (url.pathname.startsWith('/thumb/')) e.respondWith(fromPack(req, false));
});

async function page(req, e) {
    const net = fetch(req);
    // the server's copy is kept for next time, also when the device's was shown meanwhile
    e.waitUntil(net.then(r => {
        if (r.ok && !r.redirected && r.type === 'basic') return keepShell(r.clone());
    }).catch(() => {}));
    const stored = await caches.match('/', { cacheName: SHELL });
    if (!stored) return net;
    // a slow connection: after SLOW_PAGE ms the copy on the device (the same app, this deploy's or
    // the one before; the page then fetches its data from the server as always)
    const late = new Promise(res => setTimeout(() => res(stored), SLOW_PAGE));
    try {
        return await Promise.race([net, late]);
    } catch (err) {
        return stored;
    }
}

// the html, and of the static files only the versions it names: a stored shell must find the
// very modules it was made with, never a newer lib.js under an older app.js
async function keepShell(r) {
    const text = await r.clone().text();
    await (await caches.open(SHELL)).put('/', r);
    const named = new Set([...text.matchAll(/"(\/static\/[^"?]+\?v=[^"]+)"/g)].map(m => m[1]));
    const paths = new Set([...named].map(u => u.split('?')[0]));
    const c = await caches.open(STATIC);
    for (const k of await c.keys()) {
        const u = new URL(k.url);
        if (u.searchParams.has('v') && paths.has(u.pathname) && !named.has(u.pathname + u.search)) await c.delete(k);
    }
}

async function staticFile(req, url, e) {
    const c = await caches.open(STATIC);
    const hit = await c.match(req, { ignoreVary: true });
    if (hit) return hit;
    const r = await fetch(req);
    if (r.ok && r.type === 'basic') {
        const copy = r.clone();
        e.waitUntil(c.put(req, copy).catch(() => {}));
    }
    return r;
}

// shrink.js's pack(), undone: 'RSPDF1', uint32 header length, JSON header, LZMA of (0, the PDF with eexec parts decrypted)
async function unpack(r) {
    const b = new Uint8Array(await r.arrayBuffer());
    const n = new DataView(b.buffer).getUint32(6);
    const head = JSON.parse(new TextDecoder().decode(b.subarray(10, 10 + n)));
    const plain = await new Promise((res, rej) => self.LZMA.decompress(b.subarray(10 + n), (x, err) => (err ? rej(err) : res(x))));
    const pdf = Uint8Array.from(plain).subarray(1);
    for (const [at, len] of head.eexec.slice().reverse()) {
        let k = 55665;
        for (let i = at; i < at + len; i++) {
            const c = pdf[i] ^ (k >> 8);
            k = ((c + k) * 52845 + 22719) & 0xffff;
            pdf[i] = c;
        }
    }
    return new Response(pdf, { headers: { 'Content-Type': 'application/pdf', 'Content-Length': String(pdf.length) } });
}

async function fromPack(req, deviceFirst) {
    const stored = async () => {
        const hit = await caches.match(req, { cacheName: PACK, ignoreVary: true, ignoreSearch: true });
        return hit && hit.headers.get(PACKED) ? unpack(hit) : hit;
    };
    // cache: 'no-store' asks for the server's copy only (offline.js getting the full PDF, which must never
    // turn out to be the light copy kept on the device), offline too
    if (req.cache === 'no-store') return fetch(req);
    if (deviceFirst) {
        const hit = await stored();
        if (hit) return hit;
    }
    try {
        return await fetch(req);
    } catch (err) {
        const hit = await stored();
        if (hit) return hit;
        throw err;
    }
}
