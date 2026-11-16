// the service worker (served as /sw.js, registered by app.js): the site opens without a network,
// on a plane. it never changes what a page gets while there is a network: pages and data always
// come from the server; only when the server can't be reached does it answer from the device.
//   pages (any address of the app): the server's, a copy kept; offline, that copy (the html is
//     the same for every address)
//   versioned static files (?v=, vendor/): from the device once fetched (they never change
//     under their url), older versions of the same file dropped
//   the reading pack (static/app/offline.js writes it into PACK): PDFs from the device first,
//     thumbnails from the server, else the device
// a broken service worker could keep the site from loading: keep this small. to take it off
// everyone's browsers, serve one whose install calls self.registration.unregister()
const SHELL = 'shell-v1';
const STATIC = 'static-v1';
const PACK = 'pack-v1'; // as in offline.js
const APP_PATHS = ['/', '/digest', '/profile', '/import', '/inspect', '/stats', '/agents', '/graph', '/offline', '/about', '/privacy', '/imprint'];

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
    try {
        const r = await fetch(req);
        if (r.ok && !r.redirected && r.type === 'basic') {
            const copy = r.clone();
            e.waitUntil(caches.open(SHELL).then(c => c.put('/', copy)).catch(() => {}));
        }
        return r;
    } catch (err) {
        const r = await caches.match('/', { cacheName: SHELL });
        if (r) return r;
        throw err;
    }
}

async function staticFile(req, url, e) {
    const c = await caches.open(STATIC);
    const hit = await c.match(req, { ignoreVary: true });
    if (hit) return hit;
    const r = await fetch(req);
    if (r.ok && r.type === 'basic') {
        const copy = r.clone();
        e.waitUntil((async () => {
            await c.put(req, copy);
            for (const k of await c.keys()) {
                const u = new URL(k.url);
                if (u.pathname === url.pathname && u.search !== url.search) await c.delete(k);
            }
        })().catch(() => {}));
    }
    return r;
}

async function fromPack(req, deviceFirst) {
    const stored = () => caches.match(req, { cacheName: PACK, ignoreVary: true, ignoreSearch: true });
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
