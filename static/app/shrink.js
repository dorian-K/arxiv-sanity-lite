// a lighter copy of a PDF, made on the device (offline.js, for papers that leave the offline set):
// the text, fonts and drawings stay exactly as they are, the raster images (most of a paper's
// megabytes: photos, screenshots, plots saved as PNG) are scaled down and stored as JPEG. images it
// can't read safely (CMYK, JPEG 2000, 16-bit, colour-key masks, unusual colour spaces) stay untouched.
// loaded only when needed: pdf-lib is ~0.5 MB
import { PDFDocument, PDFName, PDFRawStream, PDFArray, PDFDict, PDFNumber, PDFRef, PDFString, PDFHexString, decodePDFRawStream } from 'pdf-lib';

export const MAX_SIDE = 1000; // px, the longer side of an image after shrinking
const QUALITY = 0.6; // JPEG
const MIN_BYTES = 12000; // smaller images stay as they are
const MAX_PIXELS = 40e6; // bigger ones aren't decoded (a phone's memory)

const N = PDFName.of;
const tick = () => new Promise(r => setTimeout(r, 0));

const num = (dict, key) => {
    const v = dict.lookup(N(key));
    return v instanceof PDFNumber ? v.asNumber() : undefined;
};

// the stream's single filter (null: none, undefined: one we don't take, or a chain of them)
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

// {comps, palette?} of a colour space: 1 (gray) or 3 (RGB) components, an Indexed one with its
// palette in its base space; null for anything else (CMYK, Lab, Separation, ...)
function colorSpace(cs, indexedOk = true) {
    if (cs instanceof PDFName) {
        return cs === N('DeviceRGB') ? { comps: 3 } : cs === N('DeviceGray') ? { comps: 1 } : null;
    }
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
        else if (lookup instanceof PDFRawStream) bytes = decodePDFRawStream(lookup).decode();
        return base && bytes ? { comps: 1, palette: { comps: base.comps, bytes } } : null;
    }
    return null;
}

// PNG predictors (DecodeParms /Predictor 10..15), in place of a copy
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

// the image's pixels as ImageData, or null if it isn't one we read
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

// something drawImage takes (ImageBitmap), or null
async function bitmap(stream, w, h) {
    const f = filterOf(stream.dict);
    if (f === N('DCTDecode')) {
        const cs = colorSpace(stream.dict.lookup(N('ColorSpace')), false);
        if (!cs) return null; // CMYK JPEGs come out inverted or not at all
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
    return [c, ctx];
}

// a soft mask (transparency) at the new size, as 8-bit gray: Flate, lossless, so edges stay clean
async function shrinkMask(doc, stream, w, h) {
    const { dict } = stream;
    if (dict.lookup(N('Matte')) || dict.lookup(N('Decode'))) return null;
    const ow = num(dict, 'Width'), oh = num(dict, 'Height');
    if (!ow || !oh || ow * oh > MAX_PIXELS) return null;
    const bmp = await bitmap(stream, ow, oh);
    if (!bmp) return null;
    const [, ctx] = draw(bmp, w, h);
    bmp.close && bmp.close();
    const px = ctx.getImageData(0, 0, w, h).data;
    const gray = new Uint8Array(w * h);
    for (let i = 0; i < gray.length; i++) gray[i] = px[i * 4];
    return doc.context.flateStream(gray, {
        Type: 'XObject', Subtype: 'Image', Width: w, Height: h, ColorSpace: 'DeviceGray', BitsPerComponent: 8,
    });
}

// bytes of the lighter PDF, or null when it isn't worth it (nothing to shrink) or can't be read.
// progress(done, total) after each image
export async function shrinkPdf(bytes, progress = () => {}) {
    let doc;
    try {
        doc = await PDFDocument.load(bytes, { updateMetadata: false });
    } catch (e) {
        return null; // encrypted, or broken beyond what pdf-lib reads
    }
    const { context } = doc;
    const images = context.enumerateIndirectObjects()
        .filter(([, o]) => o instanceof PDFRawStream && o.dict.lookup(N('Subtype')) === N('Image'));
    // masks belong to their image: shrunk with it, never on their own; a mask two images share stays
    const maskUses = new Map();
    for (const [, o] of images) {
        for (const k of ['SMask', 'Mask']) {
            const m = o.dict.get(N(k));
            if (m instanceof PDFRef) maskUses.set(m.toString(), (maskUses.get(m.toString()) || 0) + 1);
        }
    }
    let changed = 0;
    for (const [i, [ref, img]] of images.entries()) {
        progress(i, images.length);
        await tick();
        try {
            if (maskUses.has(ref.toString()) || img.contents.length < MIN_BYTES) continue;
            const { dict } = img;
            if (dict.lookup(N('ImageMask')) || dict.lookup(N('Decode')) || dict.lookup(N('Mask'))) continue;
            const w = num(dict, 'Width'), h = num(dict, 'Height');
            if (!w || !h || w * h > MAX_PIXELS) continue;
            const smRef = dict.get(N('SMask'));
            if (smRef && (!(smRef instanceof PDFRef) || maskUses.get(smRef.toString()) > 1)) continue;
            const s = Math.min(1, MAX_SIDE / Math.max(w, h));
            const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
            const bmp = await bitmap(img, w, h);
            if (!bmp) continue;
            const [c] = draw(bmp, nw, nh);
            bmp.close && bmp.close();
            const jpeg = new Uint8Array(await (await toBlob(c, 'image/jpeg', QUALITY)).arrayBuffer());
            let mask = null;
            if (smRef) {
                const sm = context.lookup(smRef);
                if (!(sm instanceof PDFRawStream)) continue;
                mask = await shrinkMask(doc, sm, nw, nh);
                if (!mask) continue;
            }
            const before = img.contents.length + (smRef ? context.lookup(smRef).contents.length : 0);
            if (jpeg.length + (mask ? mask.contents.length : 0) > before * 0.8) continue; // not worth a lossy copy
            const nd = dict.clone(context);
            nd.set(N('Width'), PDFNumber.of(nw));
            nd.set(N('Height'), PDFNumber.of(nh));
            nd.set(N('ColorSpace'), N('DeviceRGB'));
            nd.set(N('BitsPerComponent'), PDFNumber.of(8));
            nd.set(N('Filter'), N('DCTDecode'));
            nd.set(N('Length'), PDFNumber.of(jpeg.length));
            nd.delete(N('DecodeParms'));
            context.assign(ref, PDFRawStream.of(nd, jpeg));
            if (mask) context.assign(smRef, mask);
            changed++;
        } catch (e) {
            // this image stays as it was
        }
    }
    progress(images.length, images.length);
    if (!changed) return null;
    const out = await doc.save({ useObjectStreams: true, updateFieldAppearances: false });
    return out.length < bytes.byteLength * 0.9 ? out : null;
}
