const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.csv': 'text/csv', '.json': 'application/json', '.md': 'text/plain' };
http.createServer((req, res) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); } catch { res.writeHead(400); res.end('Bad request'); return; }
  const filename = path.resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
  if (!filename.startsWith(root + path.sep) || pathname.split('/').some(p => p.startsWith('.'))) { res.writeHead(403); res.end('Forbidden'); return; }
  fs.readFile(filename, (err, body) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': (mime[path.extname(filename)] || 'application/octet-stream') + '; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(body);
  });
}).listen(Number(process.env.PORT || 4173), '0.0.0.0', () => console.log('Nucleus Lab: http://localhost:' + (process.env.PORT || 4173)));
