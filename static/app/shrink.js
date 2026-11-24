// a light copy of a PDF, made on the device (offline.js, for papers that leave the offline set),
// aiming at TARGET bytes as stored. it is stored packed (see pack(): the PDF with its streams
// uncompressed and its Type 1 fonts decrypted, LZMA-compressed as a whole, which takes a third or
// less of what the same PDF does; sw.js unpacks it when it is opened). it goes step by step and
// stops at the first that is small enough:
//   1. what no reader needs goes (XMP metadata, the info dicts of included figures, which can hold
//      a whole draw.io source, tagging, colour profiles, embedded files); raster images (photos,
//      screenshots, plots saved as PNG) are scaled down and stored as JPEG
//   2. text fonts are left out and drawn with the reader's own Times, Helvetica or Courier (the 14
//      standard fonts every PDF reader has; the widths stay, so lines and spacing don't move). math
//      and symbol fonts stay: a stand-in would turn the symbols into wrong letters
//   3. images smaller still (400, then 250 px)
//   4. images become gray boxes
//   5. big vector drawings become gray boxes, text fonts with glyphs beyond Latin (Δ, arrows) go too,
//      and if that isn't enough,
//      a long paper keeps its first pages, as many as fit, and a last page saying what was left out
//      (appendices go first, then the references)
// text and layout of the pages kept never change. images it can't read safely (CMYK, JPEG 2000, 16-bit,
// colour-key masks) only go in step 4. loaded only when needed: pdf-lib is ~0.5 MB
import { PDFDocument, StandardFonts, PDFName, PDFRawStream, PDFArray, PDFDict, PDFNumber, PDFRef, PDFString, PDFHexString, decodePDFRawStream } from 'pdf-lib';

export const TARGET = 200e3;
const STEPS = [
    { side: 800, quality: 0.5 },
    { side: 800, quality: 0.5, fonts: true },
    { side: 400, quality: 0.35, fonts: true },
    { side: 250, quality: 0.3, fonts: true },
    { side: 0, fonts: true },
    { side: 0, fonts: 'all', drawings: true, pages: true },
];
const MIN_BYTES = 8000; // smaller images stay as they are
const MAX_PIXELS = 40e6; // bigger ones aren't decoded (a phone's memory), only boxed in step 4
const BIG_DRAWING = 120e3; // bytes of a vector drawing (form xobject, uncompressed) that step 5 boxes
const GUESS = 0.85; // packed size / gzip size until one packing has measured it (LZMA does better)
const LZMA_URL = '/static/vendor/lzma-2.3.2/src/lzma_worker-min.js'; // a classic script: sets window.LZMA
const MAGIC = 'RSPDF1'; // as in sw.js
const GRAY = 0.9; // the boxes

const N = PDFName.of;
const tick = () => new Promise(r => setTimeout(r, 0));
const num = (dict, key) => {
    const v = dict.lookup(N(key));
    return v instanceof PDFNumber ? v.asNumber() : undefined;
};
const isStream = o => o instanceof PDFRawStream;

// -----------------------------------------------------------------------------
// what no reader needs, and the objects nothing points to any more

const DROP_KEYS = ['Metadata', 'PieceInfo', 'Thumb', 'AF', 'OutputIntents', 'StructTreeRoot', 'SpiderInfo', 'Threads'];

function strip(doc) {
    const { context } = doc;
    for (const [, o] of context.enumerateIndirectObjects()) {
        const d = isStream(o) ? o.dict : o instanceof PDFDict ? o : null;
        if (!d) continue;
        for (const k of d.keys()) {
            const name = k.decodeText();
            if (DROP_KEYS.includes(name) || name.startsWith('PTEX.')) d.delete(k);
        }
    }
    const names = doc.catalog.lookup(N('Names'));
    if (names instanceof PDFDict) names.delete(N('EmbeddedFiles'));
    const info = context.lookup(context.trailerInfo.Info);
    if (info instanceof PDFDict) for (const k of info.keys()) if (k !== N('Title') && k !== N('Author')) info.delete(k);
}

function collectGarbage(doc) {
    const { context } = doc;
    const seen = new Set();
    const stack = [context.trailerInfo.Root, context.trailerInfo.Info].filter(Boolean);
    while (stack.length) {
        const o = stack.pop();
        if (o instanceof PDFRef) {
            if (seen.has(o)) continue;
            seen.add(o);
            const t = context.lookup(o);
            if (t) stack.push(t);
        } else if (o instanceof PDFDict) stack.push(...o.values());
        else if (o instanceof PDFArray) stack.push(...o.asArray());
        else if (isStream(o)) stack.push(o.dict);
    }
    for (const [ref] of context.enumerateIndirectObjects()) if (!seen.has(ref)) context.delete(ref);
}

// -----------------------------------------------------------------------------
// images

function filterOf(dict) {
    let f = dict.lookup(N('Filter'));
    if (f instanceof PDFArray) f = f.size() === 1 ? f.lookup(0) : undefined;
    if (f === undefined) return null;
    return f instanceof PDFName ? f : undefined;
}

function decodeParms(dict) {
    let p = dict.lookup(N('DecodeParms'));
    if (p instanceof PDFArray) p = p.size() === 1 ? p.lookup(0) : undefined;
    return p instanceof PDFDict ? p : null;
}

// {comps, palette?}: 1 (gray) or 3 (RGB) components, or Indexed over one of them; null for the rest
function colorSpace(cs, indexedOk = true) {
    if (cs instanceof PDFName) return cs === N('DeviceRGB') ? { comps: 3 } : cs === N('DeviceGray') ? { comps: 1 } : null;
    if (!(cs instanceof PDFArray) || !cs.size()) return null;
    const kind = cs.lookup(0);
    if (kind === N('ICCBased')) {
        const s = cs.lookup(1);
        const n = s && s.dict ? num(s.dict, 'N') : undefined;
        return n === 1 || n === 3 ? { comps: n } : null;
    }
    if (kind === N('CalRGB')) return { comps: 3 };
    if (kind === N('CalGray')) return { comps: 1 };
    if (kind === N('Indexed') && indexedOk) {
        const base = colorSpace(cs.lookup(1), false);
        const lookup = cs.lookup(3);
        let bytes = null;
        if (lookup instanceof PDFString || lookup instanceof PDFHexString) bytes = lookup.asBytes();
        else if (isStream(lookup)) bytes = decodePDFRawStream(lookup).decode();
        return base && bytes ? { comps: 1, palette: { comps: base.comps, bytes } } : null;
    }
    return null;
}

// PNG predictors (DecodeParms /Predictor 10..15)
function unpredict(data, parms, w, h, comps, bpc) {
    const pred = parms ? num(parms, 'Predictor') || 1 : 1;
    if (pred === 1) return data;
    if (pred < 10) return null; // TIFF predictor: rare, left alone
    const colors = num(parms, 'Colors') || comps;
    const bits = num(parms, 'BitsPerComponent') || bpc;
    const cols = num(parms, 'Columns') || w;
    const bpp = Math.max(1, Math.ceil((colors * bits) / 8));
    const row = Math.ceil((colors * bits * cols) / 8);
    if (data.length < (row + 1) * h) return null;
    const out = new Uint8Array(row * h);
    for (let y = 0; y < h; y++) {
        const type = data[y * (row + 1)];
        const src = y * (row + 1) + 1, dst = y * row, up = dst - row;
        for (let x = 0; x < row; x++) {
            const a = x >= bpp ? out[dst + x - bpp] : 0;
            const b = y ? out[up + x] : 0;
            const c = x >= bpp && y ? out[up + x - bpp] : 0;
            let v = data[src + x];
            if (type === 1) v += a;
            else if (type === 2) v += b;
            else if (type === 3) v += (a + b) >> 1;
            else if (type === 4) {
                const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
                v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
            } else if (type !== 0) return null;
            out[dst + x] = v & 255;
        }
    }
    return out;
}

function pixels(stream, w, h) {
    const { dict } = stream;
    const bpc = num(dict, 'BitsPerComponent');
    const cs = colorSpace(dict.lookup(N('ColorSpace')));
    if (!cs || !bpc) return null;
    if (cs.palette ? ![1, 2, 4, 8].includes(bpc) : bpc !== 8) return null;
    let data = decodePDFRawStream(stream).decode();
    data = unpredict(data, decodeParms(dict), w, h, cs.comps, bpc);
    if (!data) return null;
    const row = Math.ceil((cs.comps * bpc * w) / 8);
    if (data.length < row * h) return null;
    const img = new ImageData(w, h);
    const px = img.data;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const o = (y * w + x) * 4;
            let r, g, b;
            if (cs.palette) {
                const bit = x * bpc;
                const i = (data[y * row + (bit >> 3)] >> (8 - bpc - (bit & 7))) & ((1 << bpc) - 1);
                const p = cs.palette;
                if (p.comps === 3) { r = p.bytes[i * 3]; g = p.bytes[i * 3 + 1]; b = p.bytes[i * 3 + 2]; } else r = g = b = p.bytes[i];
            } else if (cs.comps === 3) {
                const i = y * row + x * 3;
                r = data[i]; g = data[i + 1]; b = data[i + 2];
            } else r = g = b = data[y * row + x];
            px[o] = r ?? 0; px[o + 1] = g ?? 0; px[o + 2] = b ?? 0; px[o + 3] = 255;
        }
    }
    return img;
}

async function bitmap(stream, w, h) {
    const f = filterOf(stream.dict);
    if (f === N('DCTDecode')) {
        if (!colorSpace(stream.dict.lookup(N('ColorSpace')), false)) return null; // CMYK JPEGs come out wrong
        return createImageBitmap(new Blob([stream.contents], { type: 'image/jpeg' }));
    }
    if (f !== null && f !== N('FlateDecode') && f !== N('LZWDecode') && f !== N('RunLengthDecode')) return null;
    const img = pixels(stream, w, h);
    return img && createImageBitmap(img);
}

function canvas(w, h) {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
}

const toBlob = (c, type, quality) => (c.convertToBlob
    ? c.convertToBlob({ type, quality })
    : new Promise((res, rej) => c.toBlob(b => (b ? res(b) : rej(new Error('no ' + type))), type, quality)));

function draw(bmp, w, h) {
    const c = canvas(w, h);
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bmp, 0, 0, w, h);
    if (bmp.close) bmp.close();
    return [c, ctx];
}

async function shrinkMask(doc, stream, w, h) {
    const { dict } = stream;
    if (dict.lookup(N('Matte')) || dict.lookup(N('Decode'))) return null;
    const ow = num(dict, 'Width'), oh = num(dict, 'Height');
    if (!ow || !oh || ow * oh > MAX_PIXELS) return null;
    const bmp = await bitmap(stream, ow, oh);
    if (!bmp) return null;
    const [, ctx] = draw(bmp, w, h);
    const px = ctx.getImageData(0, 0, w, h).data;
    const gray = new Uint8Array(w * h);
    for (let i = 0; i < gray.length; i++) gray[i] = px[i * 4];
    return doc.context.flateStream(gray, {
        Type: 'XObject', Subtype: 'Image', Width: w, Height: h, ColorSpace: 'DeviceGray', BitsPerComponent: 8,
    });
}

const imagesOf = context => context.enumerateIndirectObjects()
    .filter(([, o]) => isStream(o) && o.dict.lookup(N('Subtype')) === N('Image'));

// masks belong to their image: shrunk with it, never on their own (a shared one stays)
function maskUses(images) {
    const uses = new Map();
    for (const [, o] of images) {
        for (const k of ['SMask', 'Mask']) {
            const m = o.dict.get(N(k));
            if (m instanceof PDFRef) uses.set(m, (uses.get(m) || 0) + 1);
        }
    }
    return uses;
}

// scaled to at most `side` px and stored as JPEG, where that makes it smaller
async function shrinkImages(doc, side, quality, say) {
    const { context } = doc;
    const images = imagesOf(context);
    const uses = maskUses(images);
    for (const [i, [ref, img]] of images.entries()) {
        say(i, images.length);
        await tick();
        try {
            if (uses.has(ref) || img.contents.length < MIN_BYTES) continue;
            const { dict } = img;
            if (dict.lookup(N('ImageMask')) || dict.lookup(N('Decode')) || dict.lookup(N('Mask'))) continue;
            const w = num(dict, 'Width'), h = num(dict, 'Height');
            if (!w || !h || w * h > MAX_PIXELS) continue;
            const smRef = dict.get(N('SMask'));
            if (smRef && (!(smRef instanceof PDFRef) || uses.get(smRef) > 1)) continue;
            const s = Math.min(1, side / Math.max(w, h));
            const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
            if (s === 1 && filterOf(dict) === N('DCTDecode') && quality >= 0.5) continue; // already a small JPEG
            const bmp = await bitmap(img, w, h);
            if (!bmp) continue;
            const [c] = draw(bmp, nw, nh);
            const jpeg = new Uint8Array(await (await toBlob(c, 'image/jpeg', quality)).arrayBuffer());
            let mask = null;
            if (smRef) {
                const sm = context.lookup(smRef);
                if (!isStream(sm)) continue;
                mask = await shrinkMask(doc, sm, nw, nh);
                if (!mask) continue;
            }
            const before = img.contents.length + (smRef ? context.lookup(smRef).contents.length : 0);
            if (jpeg.length + (mask ? mask.contents.length : 0) > before * 0.8) continue;
            const nd = dict.clone(context);
            nd.set(N('Width'), PDFNumber.of(nw));
            nd.set(N('Height'), PDFNumber.of(nh));
            nd.set(N('ColorSpace'), N('DeviceRGB'));
            nd.set(N('BitsPerComponent'), PDFNumber.of(8));
            nd.set(N('Filter'), N('DCTDecode'));
            nd.delete(N('DecodeParms'));
            context.assign(ref, PDFRawStream.of(nd, jpeg));
            if (mask) context.assign(smRef, mask);
        } catch (e) {
            // this image stays as it was
        }
    }
}

// every image a light gray box (a 1x1 image is stretched over the image's place); stencil masks paint nothing
function boxImages(doc) {
    const { context } = doc;
    for (const [ref, img] of imagesOf(context)) {
        const stencil = img.dict.lookup(N('ImageMask'));
        const d = context.obj({ Type: 'XObject', Subtype: 'Image', Width: 1, Height: 1, BitsPerComponent: stencil ? 1 : 8 });
        if (stencil) d.set(N('ImageMask'), img.dict.lookup(N('ImageMask')));
        else d.set(N('ColorSpace'), N('DeviceGray'));
        context.assign(ref, PDFRawStream.of(d, new Uint8Array([stencil ? 0xff : Math.round(GRAY * 255)])));
    }
}

// -----------------------------------------------------------------------------
// fonts

// math and symbol fonts: kept (names after the subset prefix ABCDEF+)
const MATH_FONT = /^(cm(sy|mi|ex|bsy|mib|bsy)|eu[frsx]|ms(am|bm)|rsfs|stmary|wasy|lasy|lm(math|mathsymbols|mathitalic|mathextension)|latinmodernmath|stix.*(math|size|integral|variant|nonuni)|stixgeneral|xits|(n|r)?(tx|px|zx)(sy|ex|mi|sys|exx|mia)|mnsymbol|symbol|dingbat|zapf|bbold|bbm|dsrom|esint|cambria.?math|asana|texgyre.*math|libertinusmath|fourier|futm|mathematicalpi|mt(sy|ex|mi)|cmmi|msbm|wingding|webding)/i;
const MONO = /mono|courier|cmtt|cmsltt|^tt|inconsolata|consol|typewriter|code|nimbusmon|lmmono|sftt|txtt|menlo/i;
const SANS = /sans|arial|helvetica|verdana|calibri|myriad|segoe|tahoma|roboto|lato|futura|gill|cmss|lmsans|nimbussan|avenir|frutiger|univers|inter\b|dejavusans|liberationsans/i;
const BOLD = /bold|medi|semibold|demi|heavy|black|cmbx|cmb\d|bx\d/i;
const ITALIC = /ital|oblique|slant|cmti|cmsl|cmmi|-it\b|-it[a-z]|kursiv|ti\d+$/i;

// glyphs the standard Times, Helvetica and Courier have (Adobe's standard Latin set, and TeX's ff ligatures,
// which the readers draw fine); a text font using any other (CMR's Greek capitals in math, arrows) stays
const LATIN = new RegExp('^([A-Za-z]|[A-Za-z]{1,2}(acute|grave|circumflex|dieresis|tilde|ring|cedilla|caron|breve|macron|ogonek|dotaccent|hungarumlaut|commaaccent)|' +
    'zero|one|two|three|four|five|six|seven|eight|nine|space|nbspace|exclam|quotedbl|numbersign|dollar|percent|ampersand|' +
    'quoteright|quoteleft|quotesingle|parenleft|parenright|asterisk|plus|comma|hyphen|period|slash|colon|semicolon|less|' +
    'equal|greater|question|at|bracketleft|backslash|bracketright|asciicircum|underscore|grave|braceleft|bar|braceright|' +
    'asciitilde|exclamdown|cent|sterling|fraction|yen|florin|section|currency|quotedblleft|quotedblright|quotesinglbase|' +
    'quotedblbase|guillemotleft|guillemotright|guilsinglleft|guilsinglright|fi|fl|ff|ffi|ffl|endash|emdash|dagger|' +
    'daggerdbl|periodcentered|paragraph|bullet|ellipsis|perthousand|questiondown|acute|circumflex|tilde|macron|breve|' +
    'dotaccent|dieresis|ring|cedilla|hungarumlaut|ogonek|caron|AE|ae|OE|oe|Oslash|oslash|Lslash|lslash|dotlessi|' +
    'germandbls|Eth|eth|Thorn|thorn|copyright|registered|trademark|degree|plusminus|multiply|divide|minus|logicalnot|' +
    'mu|onehalf|onequarter|threequarters|onesuperior|twosuperior|threesuperior|ordfeminine|ordmasculine|brokenbar|' +
    'visiblespace|cwm|compwordmark|\\.notdef)$');

// the glyphs a simple font has: its subset's CharSet, else every name its encoding gives
function glyphNames(font, fd) {
    const cs = fd.lookup(N('CharSet'));
    if (cs instanceof PDFString || cs instanceof PDFHexString) return cs.decodeText().split('/').filter(Boolean);
    const enc = font.lookup(N('Encoding'));
    return enc instanceof PDFDict ? [...differences(enc).values()] : [];
}

const STD = {
    serif: ['Times-Roman', 'Times-Bold', 'Times-Italic', 'Times-BoldItalic'],
    sans: ['Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Helvetica-BoldOblique'],
    mono: ['Courier', 'Courier-Bold', 'Courier-Oblique', 'Courier-BoldOblique'],
};

// code -> glyph name of a Type 1 font program's own encoding (its clear-text part), or 'StandardEncoding'
function builtinEncoding(file) {
    const data = decodePDFRawStream(file).decode();
    const clear = num(file.dict, 'Length1') || data.length;
    let text = '';
    for (let i = 0; i < Math.min(clear, data.length); i++) text += String.fromCharCode(data[i]);
    const end = text.indexOf('eexec');
    if (end > 0) text = text.slice(0, end);
    if (/\/Encoding\s+StandardEncoding\s+def/.test(text)) return 'StandardEncoding';
    const enc = new Map();
    for (const m of text.matchAll(/dup\s+(\d+)\s*\/([^\s{}[\]()<>/%]+)\s+put/g)) enc.set(Number(m[1]), m[2]);
    return enc.size ? enc : null;
}

function differences(dict) {
    const out = new Map();
    const diffs = dict.lookup(N('Differences'));
    if (!(diffs instanceof PDFArray)) return out;
    let code = 0;
    for (const v of diffs.asArray()) {
        if (v instanceof PDFNumber) code = v.asNumber();
        else if (v instanceof PDFName) out.set(code++, v.decodeText());
    }
    return out;
}

// the font's encoding written out in the font dict, so readers know its glyphs without the program;
// false if it can't be known
function spellOutEncoding(context, font, file, kind) {
    const enc = font.lookup(N('Encoding'));
    if (enc instanceof PDFName) return true;
    if (enc instanceof PDFDict && enc.lookup(N('BaseEncoding'))) return true;
    if (kind !== 'FontFile') return false; // TrueType and CFF programs: their own encoding isn't read here
    const built = builtinEncoding(file);
    if (!built) return false;
    if (built === 'StandardEncoding') {
        if (!(enc instanceof PDFDict)) font.set(N('Encoding'), N('StandardEncoding'));
        else enc.set(N('BaseEncoding'), N('StandardEncoding'));
        return true;
    }
    const all = new Map(built);
    if (enc instanceof PDFDict) for (const [c, n] of differences(enc)) all.set(c, n);
    const arr = context.obj([]);
    for (const [c, n] of [...all].sort((a, b) => a[0] - b[0])) { arr.push(PDFNumber.of(c)); arr.push(N(n)); }
    font.set(N('Encoding'), context.obj({ Type: 'Encoding', Differences: arr }));
    return true;
}

const fontName = font => {
    const base = font.lookup(N('BaseFont'));
    return base instanceof PDFName ? base.decodeText().replace(/^[A-Z]{6}\+/, '') : '';
};
const FONT_FILES = ['FontFile', 'FontFile2', 'FontFile3', 'CIDSet', 'CharSet'];

// simple text fonts (Type 1, TrueType, CFF) without their programs, as the standard font of their
// style; composite ones (Word's, PowerPoint's) without theirs, read through their ToUnicode map
function dropTextFonts(doc, all) {
    const { context } = doc;
    for (const [, font] of context.enumerateIndirectObjects()) {
        try {
            if (!(font instanceof PDFDict) || font.lookup(N('Type')) !== N('Font')) continue;
            const sub = font.lookup(N('Subtype'));
            if (sub === N('Type0')) {
                const kids = font.lookup(N('DescendantFonts'));
                const cid = kids instanceof PDFArray ? kids.lookup(0) : null;
                const fd = cid instanceof PDFDict ? cid.lookup(N('FontDescriptor')) : null;
                if (fd instanceof PDFDict && font.lookup(N('ToUnicode')) && !MATH_FONT.test(fontName(font))) {
                    for (const k of FONT_FILES) fd.delete(N(k));
                }
                continue;
            }
            if (sub !== N('Type1') && sub !== N('TrueType') && sub !== N('MMType1')) continue;
            const fd = font.lookup(N('FontDescriptor'));
            if (!(fd instanceof PDFDict)) continue;
            const kind = ['FontFile', 'FontFile2', 'FontFile3'].find(k => fd.has(N(k)));
            if (!kind) continue;
            const name = fontName(font);
            if (!name || MATH_FONT.test(name)) continue;
            if (!spellOutEncoding(context, font, fd.lookup(N(kind)), kind)) continue;
            // Δ, ↓, ...: the standard fonts lack them, so such a font goes only when pages would go otherwise
            if (!all && glyphNames(font, fd).some(g => !LATIN.test(g))) continue;
            const flags = num(fd, 'Flags') || 0;
            const bold = BOLD.test(name) || (num(fd, 'FontWeight') || 0) >= 600;
            const italic = ITALIC.test(name) || (num(fd, 'ItalicAngle') || 0) !== 0 || (flags & 64) !== 0;
            const family = MONO.test(name) ? 'mono' : SANS.test(name) ? 'sans' : 'serif';
            const std = STD[family][(bold ? 1 : 0) + (italic ? 2 : 0)];
            font.set(N('BaseFont'), N(std));
            fd.set(N('FontName'), N(std));
            fd.set(N('Flags'), PDFNumber.of((family === 'mono' ? 1 : 0) | (family === 'serif' ? 2 : 0) | 32 | (italic ? 64 : 0)));
            for (const k of FONT_FILES) fd.delete(N(k));
        } catch (e) {
            // this font stays
        }
    }
}

// -----------------------------------------------------------------------------
// vector drawings

function boxDrawings(doc) {
    const { context } = doc;
    for (const [ref, o] of context.enumerateIndirectObjects()) {
        if (!isStream(o) || o.dict.lookup(N('Subtype')) !== N('Form') || o.contents.length < BIG_DRAWING) continue;
        const bbox = o.dict.lookup(N('BBox'));
        if (!(bbox instanceof PDFArray) || bbox.size() !== 4) continue;
        const [x0, y0, x1, y1] = bbox.asArray().map(v => (v instanceof PDFNumber ? v.asNumber() : 0));
        const d = o.dict.clone(context);
        d.set(N('Resources'), context.obj({}));
        for (const k of ['Filter', 'DecodeParms', 'Group']) d.delete(N(k));
        const ops = `q ${GRAY} g ${x0} ${y0} ${x1 - x0} ${y1 - y0} re f Q`;
        context.assign(ref, PDFRawStream.of(d, new TextEncoder().encode(ops)));
    }
}

// -----------------------------------------------------------------------------
// pages

// the first `keep` pages and a last one saying what was left out; links and bookmarks to the pages
// left out lead to that one. cut: {total}, the paper's pages
async function keepPages(doc, keep, cut) {
    const { context } = doc;
    const pages = doc.getPages();
    const gone = new Set(pages.slice(keep).map(p => p.ref));
    for (let i = pages.length - 1; i >= keep; i--) doc.removePage(i);
    const total = cut.total;
    const { width: w, height: h } = pages[0].getSize();
    const note = doc.addPage([w, h]);
    const font = await doc.embedFont(StandardFonts.Helvetica);
    const left = keep + 1 === total ? `page ${total} was` : `pages ${keep + 1} to ${total} were`;
    const lines = [
        `This light copy ends here: ${left} left out to save space on this device.`,
        'The full PDF is on the paper\'s page when you are online.',
    ];
    lines.forEach((t, i) => note.drawText(t, { x: 50, y: h - 80 - i * 18, size: 11, font, maxWidth: w - 100 }));
    // destinations are arrays [page ...], also deep inside annotations (/A << /D [...] >>) and name trees
    const redirect = o => {
        if (isStream(o)) redirect(o.dict);
        else if (o instanceof PDFDict) o.values().forEach(redirect);
        else if (o instanceof PDFArray) {
            if (o.size() && gone.has(o.get(0))) o.set(0, note.ref);
            o.asArray().forEach(redirect);
        }
    };
    for (const [, o] of context.enumerateIndirectObjects()) redirect(o);
}

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// packing: what is stored on the device. general-purpose compression over the whole file finds
// what each stream's own Flate can't (the same words and operators on every page), and Type 1
// fonts compress 3-6x better decrypted (their eexec part is encrypted, so it looks random). the
// eexec cipher is undone exactly by redoing it, so the PDF comes back byte for byte:
//   'RSPDF1', uint32 length of a JSON header {eexec: [[offset, length], ...]}, the header,
//   LZMA of (a 0 byte, which keeps lzma-js from handing back text, + the PDF with those parts decrypted)

const PLAIN_FILTERS = new Set(['FlateDecode', 'LZWDecode', 'ASCII85Decode', 'ASCIIHexDecode', 'RunLengthDecode']);

// every stream but images uncompressed (the packing compresses them better, all together)
function decodeStreams(doc) {
    const { context } = doc;
    for (const [ref, o] of context.enumerateIndirectObjects()) {
        if (!isStream(o) || o.dict.lookup(N('Subtype')) === N('Image')) continue;
        const f = o.dict.lookup(N('Filter'));
        if (!f) continue;
        const filters = f instanceof PDFArray ? f.asArray().map(x => context.lookup(x)) : [f];
        if (!filters.every(x => x instanceof PDFName && PLAIN_FILTERS.has(x.decodeText()))) continue;
        const parms = o.dict.lookup(N('DecodeParms'));
        if (parms instanceof PDFDict || parms instanceof PDFArray) continue; // predictors aren't undone by pdf-lib
        try {
            const data = decodePDFRawStream(o).decode();
            const d = o.dict.clone(context);
            d.delete(N('Filter'));
            d.delete(N('DecodeParms'));
            context.assign(ref, PDFRawStream.of(d, data));
        } catch (e) {
            // stays compressed
        }
    }
}

// Type 1 eexec: the same r feeds on the cipher byte both ways, so encrypt(decrypt(x)) is x for any bytes
function eexec(buf, decrypt) {
    let r = 55665;
    for (let i = 0; i < buf.length; i++) {
        const b = buf[i];
        const out = b ^ (r >> 8);
        r = (((decrypt ? b : out) + r) * 52845 + 22719) & 0xffff;
        buf[i] = out;
    }
}

function find(hay, needle, from = 0) {
    const first = needle[0];
    for (let i = hay.indexOf(first, from); i >= 0 && i <= hay.length - needle.length; i = hay.indexOf(first, i + 1)) {
        let j = 1;
        while (j < needle.length && hay[i + j] === needle[j]) j++;
        if (j === needle.length) return i;
    }
    return -1;
}

// {plain, regions}: what gets compressed, from the saved PDF and the document it was saved from
function prepare(pdf, doc) {
    const plain = new Uint8Array(pdf.length + 1);
    plain.set(pdf, 1);
    const regions = [];
    const taken = new Set();
    for (const [, fd] of doc.context.enumerateIndirectObjects()) {
        if (!(fd instanceof PDFDict) || fd.lookup(N('Type')) !== N('FontDescriptor')) continue;
        const file = fd.lookup(N('FontFile'));
        if (!isStream(file) || file.dict.lookup(N('Filter'))) continue;
        const l1 = num(file.dict, 'Length1'), l2 = num(file.dict, 'Length2');
        if (!l1 || !l2 || l1 + l2 > file.contents.length || l2 < 64) continue;
        const part = file.contents.subarray(l1, l1 + l2);
        let at = find(pdf, part.subarray(0, 64));
        while (at >= 0 && taken.has(at)) at = find(pdf, part.subarray(0, 64), at + 1);
        if (at < 0) continue;
        taken.add(at);
        eexec(plain.subarray(at + 1, at + 1 + l2), true);
        regions.push([at, l2]);
    }
    return { pdf, plain, regions };
}

let lzmaLoad = null;
function lzma() {
    if (globalThis.LZMA && globalThis.LZMA.compress) return Promise.resolve(globalThis.LZMA);
    lzmaLoad = lzmaLoad || new Promise((res, rej) => {
        const el = document.createElement('script');
        el.src = LZMA_URL;
        el.onload = () => res(globalThis.LZMA);
        el.onerror = () => { lzmaLoad = null; rej(new Error("the compressor didn't load")); };
        document.head.appendChild(el);
    });
    return lzmaLoad;
}

const gzipSize = async b => (await new Response(new Blob([b]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer()).byteLength;

async function pack({ plain, regions }) {
    const L = await lzma();
    const z = await new Promise((res, rej) => L.compress(plain, 1, (r, e) => (e ? rej(e) : res(r))));
    const head = new TextEncoder().encode(JSON.stringify({ eexec: regions }));
    const out = new Uint8Array(10 + head.length + z.length);
    out.set(new TextEncoder().encode(MAGIC));
    new DataView(out.buffer).setUint32(6, head.length);
    out.set(head, 10);
    out.set(Uint8Array.from(z), 10 + head.length); // lzma-js gives signed bytes
    return out;
}

export const isPacked = b => b.length > 10 && new TextDecoder().decode(b.subarray(0, 6)) === MAGIC;

// the PDF back from what pack() made
export async function unpackPdf(b) {
    const L = await lzma();
    const n = new DataView(b.buffer, b.byteOffset).getUint32(6);
    const { eexec: regions } = JSON.parse(new TextDecoder().decode(b.subarray(10, 10 + n)));
    const plain = await new Promise((res, rej) => L.decompress(b.subarray(10 + n), (r, e) => (e ? rej(e) : res(r))));
    if (typeof plain === 'string' || plain[0] !== 0) throw new Error('not a packed PDF');
    const pdf = Uint8Array.from(plain).subarray(1);
    for (const [at, len] of regions.slice().reverse()) eexec(pdf.subarray(at, at + len), false);
    return pdf;
}

// -----------------------------------------------------------------------------

// the packed light copy (the first step whose copy fits in `target`, else the smallest it got),
// or null when the PDF can't be read or it can't be packed safely. progress(text) says how far it is
export async function shrinkPdf(bytes, { target = TARGET, progress = () => {} } = {}) {
    let doc;
    try {
        doc = await PDFDocument.load(bytes, { updateMetadata: false });
    } catch (e) {
        return null; // encrypted, or broken beyond what pdf-lib reads
    }
    strip(doc);
    decodeStreams(doc);
    // the size packed, guessed from gzip (native, fast) of all but the JPEGs (which neither compresses),
    // at the ratio LZMA got over gzip on the rest, once measured
    let ratio = null;
    const save = d => d.save({ useObjectStreams: false, updateFieldAppearances: false });
    const jpegBytes = d => imagesOf(d.context).reduce((n, [, o]) => n + (filterOf(o.dict) === N('DCTDecode') ? o.contents.length : 0), 0);
    const guess = async (p, d) => {
        const j = jpegBytes(d);
        return j + Math.max(0, (await gzipSize(p.plain)) - j) * (ratio ?? GUESS);
    };
    const packed = async (p, d) => {
        progress('compressing');
        const out = await pack(p);
        const j = jpegBytes(d);
        ratio = Math.max(0.1, (out.length - j) / Math.max(1, (await gzipSize(p.plain)) - j));
        return out;
    };
    let chosen = null;
    for (const [i, step] of STEPS.entries()) {
        const last = i === STEPS.length - 1;
        const say = (done, total) => progress(`step ${i + 1} of ${STEPS.length}, image ${done} of ${total}`);
        if (step.side) await shrinkImages(doc, step.side, step.quality, say);
        else boxImages(doc);
        if (step.fonts) dropTextFonts(doc, step.fonts === 'all');
        if (step.drawings) boxDrawings(doc);
        collectGarbage(doc);
        progress(`step ${i + 1} of ${STEPS.length}, saving`);
        await tick();
        let p = prepare(await save(doc), doc);
        let pd = doc;
        let est = await guess(p, doc);
        // it might fit: measure (the guess can be off either way)
        if (est > target && est * 0.6 <= target) {
            const out = await packed(p, doc);
            if (out.length <= target) { chosen = { p, out }; break; }
            est = out.length;
        }
        if (step.pages && est > target) {
            // the most first pages that fit, found by halving (each try starts from this step's copy)
            const total = doc.getPageCount();
            let lo = 1, hi = total - 1, fit = null;
            while (lo <= hi) {
                const keep = (lo + hi) >> 1;
                progress(`step ${i + 1} of ${STEPS.length}, trying ${keep} of ${total} pages`);
                const d = await PDFDocument.load(p.pdf, { updateMetadata: false });
                await keepPages(d, keep, { total });
                collectGarbage(d);
                const tried = prepare(await save(d), d);
                if ((await guess(tried, d)) <= target * 0.97) { fit = { p: tried, d }; lo = keep + 1; } else hi = keep - 1;
            }
            if (fit) { p = fit.p; pd = fit.d; est = await guess(p, pd); }
        }
        if (est <= target || last) {
            const out = await packed(p, pd);
            chosen = { p, out };
            if (out.length <= target || last) break;
        }
    }
    // it must come back byte for byte, or it isn't kept
    const back = await unpackPdf(chosen.out);
    if (back.length !== chosen.p.pdf.length || back.some((v, k) => v !== chosen.p.pdf[k])) return null;
    return chosen.out;
}
