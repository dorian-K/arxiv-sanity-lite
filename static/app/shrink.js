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
//   3. images smaller and worse still, down to 100 px
//   from step 2, the numbers of vector drawings (plots) are rounded to 4 significant digits, from
//   the last steps to 3 (see roundNumbers), and small-caps fonts go too.
// nothing is ever left out: no page, no image, no drawing (user decision), so a paper with more
// than fits keeps its smallest copy even above TARGET.
// text and layout never change. images it can't read safely (CMYK, JPEG 2000, 16-bit, colour-key
// masks) stay as they are. loaded only when needed: pdf-lib is ~0.5 MB
import { PDFDocument, PDFName, PDFRawStream, PDFArray, PDFDict, PDFNumber, PDFRef, PDFString, PDFHexString, decodePDFRawStream } from 'pdf-lib';

export const TARGET = 200e3;
const STEPS = [
    { side: 800, quality: 0.5 },
    { side: 800, quality: 0.5, fonts: true, round: 4 },
    { side: 400, quality: 0.35, fonts: true, round: 4 },
    { side: 250, quality: 0.3, fonts: true, round: 4 },
    { side: 160, quality: 0.25, fonts: 'all', round: 3 },
    { side: 100, quality: 0.2, fonts: 'all', round: 3 },
];
const MIN_BYTES = 8000; // smaller images within the step's size stay as they are (not worth a lossy copy)
const MAX_PIXELS = 40e6; // bigger ones aren't decoded (a phone's memory) and stay as they are
const GUESS = 0.85; // packed size / gzip size until one packing has measured it (LZMA does better)
const LZMA_URL = '/static/vendor/lzma-2.3.2/src/lzma_worker-min.js'; // a classic script: sets window.LZMA
const MAGIC = 'RSPDF1'; // as in sw.js

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
            if (uses.has(ref)) continue;
            const { dict } = img;
            if (dict.lookup(N('ImageMask')) || dict.lookup(N('Decode')) || dict.lookup(N('Mask'))) continue;
            const w = num(dict, 'Width'), h = num(dict, 'Height');
            if (!w || !h || w * h > MAX_PIXELS) continue;
            const smRef = dict.get(N('SMask'));
            if (smRef && (!(smRef instanceof PDFRef) || uses.get(smRef) > 1)) continue;
            const s = Math.min(1, side / Math.max(w, h));
            const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
            if (s === 1 && (filterOf(dict) === N('DCTDecode') || img.contents.length < MIN_BYTES)) continue; // small enough already
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

const SMALL_CAPS = /csc|smallcap|caps\b|-sc\b/i;

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

// a composite font's characters, from its ToUnicode map, are all ones the standard fonts have (Latin,
// punctuation, ligatures): else it keeps its program, as simple fonts with other glyphs do (LATIN).
// a map that can't be read keeps it too
const LATIN_CODE = c => c < 0x250 || (c >= 0x2000 && c <= 0x206f) || (c >= 0xfb00 && c <= 0xfb06) || c === 0x20ac || c === 0x2122;
function latinToUnicode(font) {
    try {
        const cmap = font.lookup(N('ToUnicode'));
        if (!(cmap instanceof PDFRawStream)) return false;
        const text = new TextDecoder('latin1').decode(decodePDFRawStream(cmap).decode());
        // the destinations of bfchar and bfrange: <src> <dst>, <lo> <hi> <dst> or <lo> <hi> [<dst> ...]
        const sections = text.match(/begin(bfchar|bfrange)[\s\S]*?end\1/g) || [];
        const ok = h => {
            for (let i = 0; i + 4 <= h.length; i += 4) {
                const c = parseInt(h.slice(i, i + 4), 16);
                if ((c >= 0xd800 && c <= 0xdfff) || !LATIN_CODE(c)) return false; // surrogates: math alphanumerics, say
            }
            return true;
        };
        for (const sec of sections) {
            const t = (sec.replace(/^begin\w+/, '').match(/<[0-9A-Fa-f\s]*>|\[|\]/g) || []).map(x => x.replace(/\s/g, ''));
            if (sec.startsWith('beginbfchar')) {
                for (let k = 1; k < t.length; k += 2) if (!ok(t[k].slice(1, -1))) return false;
                continue;
            }
            for (let k = 0; k + 2 < t.length;) { // <lo> <hi> <dst> (counting up from dst) or <lo> <hi> [<dst> ...]
                const lo = parseInt(t[k].slice(1, -1), 16), hi = parseInt(t[k + 1].slice(1, -1), 16);
                if (t[k + 2] === '[') {
                    k += 3;
                    for (; k < t.length && t[k] !== ']'; k++) if (!ok(t[k].slice(1, -1))) return false;
                    k++;
                } else {
                    const d = t[k + 2].slice(1, -1);
                    const last = (parseInt(d.slice(-4), 16) + Math.max(0, hi - lo)).toString(16).padStart(4, '0');
                    if (!ok(d) || !ok(last)) return false;
                    k += 3;
                }
            }
        }
        return true;
    } catch (e) {
        return false;
    }
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
                if (fd instanceof PDFDict && latinToUnicode(font) && !MATH_FONT.test(fontName(font))) {
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
            // Δ, ≠, β: the standard fonts lack them (pdfium draws nothing, so math would be wrong): such a
            // font stays. small caps come out as spaced-out lowercase: they go only in the last steps
            if (glyphNames(font, fd).some(g => !LATIN.test(g))) continue;
            if (!all && SMALL_CAPS.test(name)) continue;
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
// vector drawings: their numbers rounded to a few significant digits. a plot of thousands of
// points is mostly coordinates written to 4-6 decimals; at 4 digits nothing visibly moves (523.456
// becomes 523.5, 0.12345 becomes 0.1235: relative, so it is as safe in a scaled-up coordinate
// system as in page points), at 3 a line may move by up to half a point

const isSpace = c => c === 32 || c === 10 || c === 13 || c === 9 || c === 12 || c === 0;
const isDelim = c => c === 40 || c === 41 || c === 60 || c === 62 || c === 91 || c === 93 || c === 123 || c === 125 || c === 47 || c === 37;
const isNumChar = c => (c >= 48 && c <= 57) || c === 46 || c === 45 || c === 43;

function roundTo(v, digits) {
    if (!v || !Number.isFinite(v)) return '0';
    const d = Math.max(0, digits - 1 - Math.floor(Math.log10(Math.abs(v))));
    let t = v.toFixed(Math.min(d, 10));
    if (t.includes('.')) t = t.replace(/0+$/, '').replace(/\.$/, '');
    return t === '-0' ? '0' : t;
}

// the content stream with its decimal numbers rounded; null if it has an inline image (binary data
// that a scan for numbers could break) or doesn't scan cleanly
function roundNumbers(data, digits) {
    const out = [];
    let i = 0, start = 0;
    const n = data.length;
    const flush = to => { if (to > start) out.push(data.subarray(start, to)); };
    while (i < n) {
        const c = data[i];
        if (isSpace(c)) { i++; continue; }
        if (c === 37) { while (i < n && data[i] !== 10 && data[i] !== 13) i++; continue; } // comment
        if (c === 40) { // (string), nested, with escapes
            let depth = 0;
            for (; i < n; i++) {
                if (data[i] === 92) { i++; continue; }
                if (data[i] === 40) depth++;
                else if (data[i] === 41 && --depth === 0) { i++; break; }
            }
            continue;
        }
        if (c === 60 && data[i + 1] !== 60) { while (i < n && data[i] !== 62) i++; i++; continue; } // <hex>
        if (isDelim(c) && c !== 47) { i++; continue; }
        // a token: a name (/...), an operator or a number
        let j = i + 1;
        while (j < n && !isSpace(data[j]) && !isDelim(data[j])) j++;
        if (c !== 47) {
            let num = true, dot = false;
            for (let k = i; k < j; k++) {
                if (!isNumChar(data[k])) { num = false; break; }
                if (data[k] === 46) dot = true;
            }
            if (num && dot) {
                const v = Number(String.fromCharCode(...data.subarray(i, j)));
                if (!Number.isNaN(v)) {
                    flush(i);
                    out.push(new TextEncoder().encode(roundTo(v, digits)));
                    start = j;
                }
            } else if (!num && j - i === 2 && data[i] === 66 && data[i + 1] === 73) return null; // BI
        }
        i = j;
    }
    flush(n);
    const len = out.reduce((a, b) => a + b.length, 0);
    const res = new Uint8Array(len);
    let at = 0;
    for (const b of out) { res.set(b, at); at += b.length; }
    return res;
}

// form xobjects (figures included as PDF: plots, diagrams), uncompressed ones only
function roundDrawings(doc, digits) {
    const { context } = doc;
    for (const [ref, o] of context.enumerateIndirectObjects()) {
        if (!isStream(o) || o.dict.lookup(N('Subtype')) !== N('Form') || o.dict.lookup(N('Filter'))) continue;
        try {
            const data = roundNumbers(o.contents, digits);
            if (data && data.length < o.contents.length) context.assign(ref, PDFRawStream.of(o.dict.clone(context), data));
        } catch (e) {
            // stays as it is
        }
    }
}

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
// 'not smaller' when no step makes it smaller than it is, or null when the PDF can't be read or it
// can't be packed safely. progress(text) says how far it is
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
        await shrinkImages(doc, step.side, step.quality, say);
        if (step.fonts) dropTextFonts(doc, step.fonts === 'all');
        if (step.round) roundDrawings(doc, step.round);
        collectGarbage(doc);
        progress(`step ${i + 1} of ${STEPS.length}, saving`);
        await tick();
        const p = prepare(await save(doc), doc);
        // packed (slow) only when the guess says it may fit (the guess can be off either way), and at the end
        if ((await guess(p, doc)) * 0.6 <= target || last) {
            const out = await packed(p, doc);
            if (!chosen || out.length < chosen.out.length) chosen = { p, out };
            if (out.length <= target) break;
        }
    }
    if (chosen.out.length >= bytes.byteLength) return 'not smaller'; // kept as it is (offline.js), not deleted
    // it must come back byte for byte, or it isn't kept
    const back = await unpackPdf(chosen.out);
    if (back.length !== chosen.p.pdf.length || back.some((v, k) => v !== chosen.p.pdf[k])) return null;
    return chosen.out;
}
