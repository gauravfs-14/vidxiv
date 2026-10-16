"""Get a paper into a job: fetch from arXiv or take an uploaded PDF, then extract text, sections and figures."""
from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf
import requests

UA = {"User-Agent": "vidxiv/0.2 (local tool; https://github.com/)"}
ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")
MAX_FIGURES = 14


def parse_arxiv_id(s: str) -> str | None:
    s = s.strip()
    s = re.sub(r"^(https?://)?(www\.)?(export\.)?arxiv\.org/(abs|pdf|html)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I).removesuffix(".pdf")
    for rx in (ARXIV_NEW, ARXIV_OLD):
        m = rx.fullmatch(s)
        if m:
            return m.group(1)
    return None


def fetch_arxiv(arxiv_id: str, dest: Path, log=print) -> dict:
    """Download the PDF and read metadata from the arXiv API. Returns {title, authors, abstract, url, published}."""
    dest.mkdir(parents=True, exist_ok=True)
    meta = {"title": "", "authors": [], "abstract": "", "url": f"https://arxiv.org/abs/{arxiv_id}", "published": ""}
    try:
        r = requests.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id}, headers=UA, timeout=30)
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ET.fromstring(r.content).find("a:entry", ns)
        if entry is not None and entry.find("a:title", ns) is not None:
            title = " ".join((entry.findtext("a:title", "", ns)).split())
            if title and title.lower() != "error":
                meta.update(
                    title=title,
                    authors=[" ".join(a.findtext("a:name", "", ns).split()) for a in entry.findall("a:author", ns)],
                    abstract=" ".join(entry.findtext("a:summary", "", ns).split()),
                    published=(entry.findtext("a:published", "", ns) or "")[:10],
                )
    except Exception as e:  # noqa: BLE001
        log(f"arXiv metadata lookup failed ({e}); using the PDF text instead")
    log(f"Downloading arxiv.org/pdf/{arxiv_id}")
    pdf = dest / "paper.pdf"
    r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=UA, timeout=90, allow_redirects=True)
    if r.status_code == 404:
        raise ValueError(f"arXiv has no paper with id {arxiv_id}")
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError("arXiv did not return a PDF (the paper may be withdrawn)")
    pdf.write_bytes(r.content)
    return meta


# ---------------------------------------------------------------- text

HEADING_WORDS = ("abstract", "introduction", "background", "related work", "preliminaries", "method", "methods", "approach",
                 "model", "experiments", "experimental setup", "results", "evaluation", "analysis", "discussion",
                 "limitations", "conclusion", "conclusions", "future work", "references", "appendix", "acknowledgments")
NUMBERED = re.compile(r"^(\d+(\.\d+){0,2}|[A-Z])\.?\s+[A-Z][^\n]{2,80}$")


def _lines_with_style(doc: pymupdf.Document):
    """Yield (page_no, text, size, bold) per text line."""
    for pno, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                size = max(s["size"] for s in spans)
                bold = all(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)
                yield pno, text, size, bold


def extract_text(doc: pymupdf.Document) -> tuple[str, str, list[dict]]:
    """Return (title_guess, markdown_text, sections). Headings are detected by size/bold/numbering."""
    rows = list(_lines_with_style(doc))
    if not rows:
        return "", "", []
    sizes: dict[float, int] = {}
    for _, t, s, _ in rows:
        sizes[round(s, 1)] = sizes.get(round(s, 1), 0) + len(t)
    body = max(sizes, key=sizes.get)
    first = [r for r in rows if r[0] == 0]
    biggest = max((r[2] for r in first), default=body)
    title = " ".join(r[1] for r in first if abs(r[2] - biggest) < 0.6)[:200] if biggest > body + 1 else ""
    out: list[str] = []
    sections: list[dict] = []
    cur = {"heading": "Front matter", "text": []}
    for _, t, s, bold in rows:
        low = t.lower().strip(" .:")
        is_head = len(t) < 90 and (
            (bold and NUMBERED.match(t)) or low in HEADING_WORDS or (s > body + 0.8 and len(t) < 70 and t != title)
        )
        if is_head and t != title:
            sections.append({"heading": cur["heading"], "text": " ".join(cur["text"])})
            cur = {"heading": t.strip(), "text": []}
            out.append(f"\n## {t.strip()}\n")
        else:
            cur["text"].append(t)
            out.append(t)
    sections.append({"heading": cur["heading"], "text": " ".join(cur["text"])})
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", "\n".join(out))
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return title, text, [s for s in sections if len(s["text"]) > 40]


def find_abstract(sections: list[dict], text: str) -> str:
    for s in sections:
        if s["heading"].lower().strip(" .:") == "abstract":
            return s["text"][:2500]
    m = re.search(r"abstract[\s.:\-—]*(.{200,2500}?)(?:\n\s*\n|\n## |1\s+Introduction|Introduction)", text, re.S | re.I)
    return " ".join(m.group(1).split()) if m else ""


# ---------------------------------------------------------------- figures

CAPTION = re.compile(r"^\s*(fig(?:ure)?\.?)\s*(\d+|[A-Z]\d+)\s*[.:|]?\s*(.*)", re.I | re.S)


def _blocks(page: pymupdf.Page):
    return [(pymupdf.Rect(b[:4]), b[4], b[6]) for b in page.get_text("blocks")]  # rect, text, type


def _figure_rect(page: pymupdf.Page, cap: pymupdf.Rect, blocks) -> pymupdf.Rect | None:
    """Grow a rectangle from the graphics nearest to a caption (above it, else below), absorbing axis labels."""
    pw, ph = page.rect.width, page.rect.height
    gfx: list[pymupdf.Rect] = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width > 20 and r.height > 20:
            gfx.append(r)
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if (r.width > 3 or r.height > 3) and r.width < pw * 0.98 and r.height < ph * 0.9:
            gfx.append(r)
    texts = [r for r, t, ty in blocks if ty == 0 and len(t.strip()) < 160 and not CAPTION.match(t)]
    for above in (True, False):
        if above:
            seeds = [r for r in gfx if r.y1 <= cap.y0 + 4 and r.y1 > cap.y0 - 90 and r.x1 > cap.x0 and r.x0 < cap.x1]
        else:
            seeds = [r for r in gfx if r.y0 >= cap.y1 - 4 and r.y0 < cap.y1 + 90 and r.x1 > cap.x0 and r.x0 < cap.x1]
        if not seeds:
            continue
        box = pymupdf.Rect(seeds[0])
        for s in seeds[1:]:
            box |= s
        for _ in range(8):
            grow = pymupdf.Rect(box.x0 - 14, box.y0 - 14, box.x1 + 14, box.y1 + 14)
            before = pymupdf.Rect(box)
            for r in gfx + texts:
                if grow.intersects(r) and (above and r.y1 <= cap.y0 + 2 or not above and r.y0 >= cap.y1 - 2):
                    box |= r
            if box == before:
                break
        box &= page.rect
        if box.width > pw * 0.2 and box.height > ph * 0.07:
            return box
    return None


def _autocrop(pix: pymupdf.Pixmap, pad: int = 14) -> pymupdf.Pixmap:
    """Trim blank margins so the figure fills its card in the video."""
    try:
        import numpy as np

        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        ys, xs = np.where((a < 244).any(axis=2))
        if len(xs) == 0:
            return pix
        x0, x1 = max(0, xs.min() - pad), min(pix.width, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(pix.height, ys.max() + pad)
        if (x1 - x0) * (y1 - y0) > 0.97 * pix.width * pix.height:
            return pix
        out = pymupdf.Pixmap(pix.colorspace, pymupdf.IRect(0, 0, x1 - x0, y1 - y0), pix.alpha)
        out.copy(pix, pymupdf.IRect(x0, y0, x1, y1))
        return out
    except Exception:  # noqa: BLE001
        return pix


def extract_figures(doc: pymupdf.Document, out_dir: Path, log=print) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    found: list[dict] = []
    seen_labels: set[str] = set()
    for pno, page in enumerate(doc):
        blocks = _blocks(page)
        for rect, text, ty in blocks:
            if ty != 0:
                continue
            m = CAPTION.match(text)
            if not m or not m.group(3).strip():
                continue
            label = f"Figure {m.group(2)}"
            if label in seen_labels:
                continue
            box = _figure_rect(page, rect, blocks)
            if box is None:
                continue
            seen_labels.add(label)
            fid = f"fig-{len(found) + 1:02d}"
            pix = page.get_pixmap(clip=box, dpi=200)
            if pix.width > 2400:
                pix = page.get_pixmap(clip=box, dpi=int(200 * 2400 / pix.width))
            pix = _autocrop(pix)
            pix.save(out_dir / f"{fid}.png")
            caption = " ".join(m.group(3).split())[:300]
            found.append({"id": fid, "file": f"{fid}.png", "page": pno + 1, "label": label, "caption": caption,
                          "width": pix.width, "height": pix.height})
            if len(found) >= MAX_FIGURES:
                return found
    if found:
        return found
    # no captioned figures: render large embedded images from the page, flattened onto white
    done: set[int] = set()
    for pno, page in enumerate(doc):
        for info in page.get_image_info(xrefs=True):
            xref, r = info.get("xref"), pymupdf.Rect(info["bbox"])
            if not xref or xref in done or info.get("width", 0) < 360 or info.get("height", 0) < 240 or r.width < 100 or r.height < 80:
                continue
            done.add(xref)
            fid = f"fig-{len(found) + 1:02d}"
            pix = _autocrop(page.get_pixmap(clip=r & page.rect, dpi=200, alpha=False))
            pix.save(out_dir / f"{fid}.png")
            found.append({"id": fid, "file": f"{fid}.png", "page": pno + 1, "label": f"Image {len(found) + 1}",
                          "caption": "", "width": pix.width, "height": pix.height})
            if len(found) >= MAX_FIGURES:
                return found
    return found


# ---------------------------------------------------------------- agent-defined regions

REGION_FILE = "ai-figures.json"
REGION_ID = re.compile(r"^x-\d{2}$")


def add_regions(job_dir: Path) -> tuple[list[dict], list[str]]:
    """Crop the regions the script agent listed in paper/ai-figures.json and add them to figures.json.

    Each region is {id: "x-01", label, kind, page (1-based), bbox: [x0, y0, x1, y1] as fractions of the page, caption}.
    Returns (figures added or updated, problems). Bad regions are reported, never fatal.
    """
    pdir = job_dir / "paper"
    src = pdir / REGION_FILE
    if not src.exists():
        return [], []
    problems: list[str] = []
    try:
        regions = json.loads(src.read_text())
        assert isinstance(regions, list)
    except (ValueError, AssertionError):
        return [], [f"{REGION_FILE} must be a JSON list of regions"]
    doc = pymupdf.open(pdir / "paper.pdf")
    figs = json.loads((pdir / "figures.json").read_text()) if (pdir / "figures.json").exists() else []
    by_id = {f["id"]: f for f in figs}
    done: list[dict] = []
    (pdir / "figures").mkdir(exist_ok=True)
    for i, r in enumerate(regions):
        rid = r.get("id", "")
        try:
            if not REGION_ID.match(rid):
                raise ValueError("id must look like x-01")
            page_no = int(r["page"])
            if not 1 <= page_no <= doc.page_count:
                raise ValueError(f"page {page_no} is outside 1 to {doc.page_count}")
            x0, y0, x1, y1 = (float(v) for v in r["bbox"])
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("bbox must be [x0, y0, x1, y1] as fractions with x0 < x1 and y0 < y1")
            if (x1 - x0) < 0.08 or (y1 - y0) < 0.04:
                raise ValueError("bbox is too small")
            page = doc[page_no - 1]
            clip = pymupdf.Rect(x0 * page.rect.width, y0 * page.rect.height, x1 * page.rect.width, y1 * page.rect.height)
            pix = _autocrop(page.get_pixmap(clip=clip, dpi=200, alpha=False))
            if pix.width > 2400:
                pix = page.get_pixmap(clip=clip, dpi=int(200 * 2400 / pix.width), alpha=False)
            pix.save(pdir / "figures" / f"{rid}.png")
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"{REGION_FILE} item {i + 1} ({rid or 'no id'}): {e}")
            continue
        entry = {"id": rid, "file": f"{rid}.png", "page": page_no, "label": str(r.get("label") or f"Page {page_no}")[:40],
                 "caption": str(r.get("caption") or "")[:300], "kind": str(r.get("kind") or "figure"), "width": pix.width, "height": pix.height}
        by_id[rid] = entry
        done.append(entry)
    (pdir / "figures.json").write_text(json.dumps(list(by_id.values()), indent=2))
    return done, problems


# ---------------------------------------------------------------- entry points

def ingest(job_dir: Path, source: dict, upload: Path | None = None, log=print) -> dict:
    """Populate job_dir/paper/. Returns the paper summary stored in job.json."""
    pdir = job_dir / "paper"
    pdir.mkdir(exist_ok=True)
    meta: dict = {"title": "", "authors": [], "abstract": "", "url": "", "published": ""}
    if source["type"] == "arxiv":
        meta.update(fetch_arxiv(source["id"], pdir, log))
    else:
        assert upload is not None
        shutil.copy(upload, pdir / "paper.pdf")
    log("Reading the PDF")
    try:
        doc = pymupdf.open(pdir / "paper.pdf")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Could not open the PDF: {e}") from e
    if doc.is_encrypted and not doc.authenticate(""):
        raise ValueError("The PDF is password protected")
    if doc.page_count == 0:
        raise ValueError("The PDF has no pages")
    title_guess, text, sections = extract_text(doc)
    if len(text.strip()) < 500:
        raise ValueError("No readable text in this PDF. It looks like a scan; run OCR on it first.")
    if not meta["title"]:
        meta["title"] = (doc.metadata or {}).get("title", "").strip() or title_guess or Path(source.get("filename", "Untitled")).stem
    if not meta["abstract"]:
        meta["abstract"] = find_abstract(sections, text)
    log("Extracting figures")
    figures = extract_figures(doc, pdir / "figures", log)
    log(f"Found {len(figures)} figure(s), {doc.page_count} pages, {len(text):,} characters")
    (pdir / "paper.md").write_text(f"# {meta['title']}\n\n" + (f"Authors: {', '.join(meta['authors'])}\n\n" if meta["authors"] else "") + text)
    (pdir / "figures.json").write_text(json.dumps(figures, indent=2))
    paper = {**meta, "pages": doc.page_count, "chars": len(text), "sections": [s["heading"] for s in sections][:40]}
    (pdir / "paper.json").write_text(json.dumps(paper, indent=2))
    return paper
