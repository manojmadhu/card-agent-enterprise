"""Generates docs/diagrams/architecture.svg and docs/diagrams/process-flow.svg.

  python tools/make_diagrams.py docs/diagrams
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
STYLES = {
    "proc":   ("#EEF3FD", "#3E6FD9", "#1B3A7A"),
    "dec":    ("#FFF5E0", "#C98A12", "#6B4708"),
    "reject": ("#FDECEC", "#C93C3C", "#7A1F1F"),
    "review": ("#FFF0E3", "#D9772B", "#7A3E10"),
    "accept": ("#E8F6EE", "#2E9D5B", "#145234"),
    "neutral": ("#F4F4F2", "#8A8A84", "#2C2C2A"),
    "svc":    ("#F1EFFD", "#6A5ACD", "#2E2570"),
    "gpu":    ("#E6F5F1", "#1D8F6E", "#0B4A39"),
    "data":   ("#F6F6F4", "#7A7A74", "#2C2C2A"),
    "ci":     ("#EAF3FB", "#2F7BBF", "#123E66"),
}


class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" font-family="{FONT}">',
            f"<title>{escape(title)}</title><desc>{escape(desc)}</desc>",
            '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M1 1L9 5L1 9z" fill="#55554F"/></marker></defs>',
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>',
        ]

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=14, weight=400, color="#2C2C2A", anchor="middle", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                 f'text-anchor="{anchor}" dominant-baseline="central"{st}>{escape(s)}</text>')

    def box(self, x, y, w, h, title, subs=(), style="proc", rx=10, dashed=False, title_size=15, sub_size=12.5,
            align="middle"):
        fill, stroke, tc = STYLES[style]
        da = ' stroke-dasharray="6 4"' if dashed else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{da}/>')
        lines = [(title, title_size, 600)] + [(s, sub_size, 400) for s in subs]
        lh = [ts * 1.45 for _, ts, _ in lines]
        cy = y + h / 2 - sum(lh) / 2 + lh[0] / 2
        tx = x + w / 2 if align == "middle" else x + 16
        for (s, ts, wt), step in zip(lines, lh):
            self.text(tx, cy, s, ts, wt, tc if wt == 600 else _mute(tc), align if align == "middle" else "start")
            cy += step

    def pill(self, x, y, w, h, title, style="neutral", sub=None):
        self.box(x, y, w, h, title, [sub] if sub else [], style, rx=h / 2)

    def diamond(self, cx, cy, hw, hh, lines, style="dec"):
        fill, stroke, tc = STYLES[style]
        self.add(f'<polygon points="{cx},{cy-hh} {cx+hw},{cy} {cx},{cy+hh} {cx-hw},{cy}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        y0 = cy - (len(lines) - 1) * 9
        for i, s in enumerate(lines):
            self.text(cx, y0 + i * 18, s, 13.5, 600 if i == 0 else 400, tc)

    def arrow(self, pts, label=None, lx=None, ly=None, dashed=False, color="#55554F", anchor="middle"):
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        da = ' stroke-dasharray="5 4"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"{da} marker-end="url(#a)"/>')
        if label:
            self.text(lx, ly, label, 12, 500, "#55554F", anchor)

    def save(self, name, h=None):
        if h:                                    # final height known only after laying out the flow
            self.parts[0] = self.parts[0].replace(f'height="{self.h}"', f'height="{h}"').replace(
                f'0 0 {self.w} {self.h}', f'0 0 {self.w} {h}')
            self.parts[3] = f'<rect width="{self.w}" height="{h}" fill="#FFFFFF"/>'
        (OUT / name).write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def _mute(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02X}{:02X}{:02X}".format(*(min(255, int(c + (255 - c) * 0.28)) for c in (r, g, b)))


# =====================================================================================
# 1) ARCHITECTURE   (matches deploy/docker-compose.yml and src/cardagent on feature/paddleocr)
# =====================================================================================
A = Svg(1500, 1200, "Card Validation Agent — architecture",
        "Components of the offline card validation agent inside a private network with a T4 GPU host.")
A.text(40, 30, "Card Validation Agent — system architecture", 22, 700, "#1F1F1D", "start")
A.text(40, 56, "All models self-hosted · no public AI APIs · containers on an internal network with no internet route",
       13.5, 400, "#6B6B66", "start")

# private network boundary
A.add('<rect x="290" y="84" width="1180" height="860" rx="18" fill="none" stroke="#9A9A94" stroke-width="1.4" stroke-dasharray="8 6"/>')
A.text(310, 106, "Private network  (on-prem / private cloud)  ·  docker network internal: true", 13, 600, "#6B6B66", "start")

# external systems
A.box(40, 250, 200, 110, "Capture system", ["POS · kiosk · mobile app", "1–2 photos: front / back"], "neutral")
A.box(40, 410, 200, 150, "Web console", ["card-agent-web (React)", "cards · references", "review queue · runs",
                                         "settings · evaluations"], "review")
A.box(40, 610, 200, 100, "Storage system", ["stores the result JSON"], "neutral")

# gateway
A.box(320, 300, 190, 220, "API gateway / nginx", ["mTLS · auth", "rate limit · WAF", "same-origin proxy",
                                                   "for the console", "(or api.cors_origins)"], "data")

# agent service container
A.add('<rect x="550" y="130" width="560" height="610" rx="16" fill="#FAF9FF" stroke="#6A5ACD" stroke-width="1.4"/>')
A.text(570, 152, "cardagent container  (FastAPI · uvicorn 2 workers · read-only FS)", 14.5, 700, "#2E2570", "start")
A.box(575, 172, 510, 112, "API layer  (/v1)", [
    "POST /cards/validate  (+ /validate/stream, SSE)  ·  /cards  ·  /registry",
    "/runs  ·  /review  ·  /evaluations  ·  /settings  ·  /stats",
    "/health/live  ·  /health/ready  ·  /metrics",
    "admin routes require X-Admin-Token"], "svc", sub_size=12)
# pipeline engine
A.add('<rect x="575" y="300" width="510" height="214" rx="10" fill="#F1EFFD" stroke="#6A5ACD" stroke-width="1.4"/>')
A.text(830, 320, "Pipeline engine  (1–2 photos per request)", 15, 600, "#2E2570")
chips = [("Quality gate", "per photo"), ("Normalise", "detect + warp"), ("OCR", "PaddleOCR 0°/180°"),
         ("Identify", "2-stage · fused"), ("Sides", "assign front/back"), ("Align", "per side"),
         ("Extract", "ROI · regex · VLM"), ("Rules → Decide", "card definition")]
for i, (t, s) in enumerate(chips):
    cx = 590 + (i % 4) * 124
    cy = 340 + (i // 4) * 84
    A.box(cx, cy, 112, 70, t, [s], "proc", rx=8, title_size=13, sub_size=11)
for i in range(3):
    for row in (0, 1):
        x = 590 + i * 124 + 112
        y = 375 + row * 84
        A.arrow([(x + 1, y), (x + 11, y)])
A.arrow([(1018, 410), (1018, 414), (646, 414), (646, 422)])
# engine state
A.box(575, 532, 510, 188, "Engine state  (in memory, per worker)", [
    "Snapshot of card definitions, references and settings overrides",
    "refreshed from Qdrant every 5 s — no restart, no reload call",
    "Keyword inverted index (EN + AR, OCR-confusable characters)",
    "Reference keypoints for stage-2 verification",
    "Idempotency cache (2 000 results)  ·  in-flight requests keep their snapshot"],
      "svc", title_size=14, sub_size=12)

# GPU host
A.add('<rect x="1150" y="130" width="290" height="440" rx="16" fill="#F3FBF8" stroke="#1D8F6E" stroke-width="1.4"/>')
A.text(1170, 152, "GPU · NVIDIA T4 16 GB (shared)", 14.5, 700, "#0B4A39", "start")
A.box(1170, 172, 250, 96, "DINOv2 ViT-S/14", ["in-process · ONNX Runtime", "image → vector (CUDA or CPU)"], "gpu")
A.box(1170, 282, 250, 130, "PaddleOCR", ["in-process · RapidOCR on ONNX Runtime", "PP-OCRv6 det + rec (bundled)",
                                         "Arabic: PP-OCRv5 rec (optional)", "Tesseract = alternative engine"], "gpu",
      sub_size=12)
A.box(1170, 426, 250, 126, "VLM fallback  (optional)", ["separate Ollama container", "qwen2.5vl:7b · off by default",
                                                          "only unreadable fields"], "gpu", dashed=True)

# Qdrant
A.box(1150, 590, 290, 150, "Qdrant v1.18.1  (API key)", ["cardagent_refs — reference vectors",
                                                          "cardagent_families — cards · settings",
                                                          "cardagent_runs — history · review",
                                                          "cardagent_evaluations"], "data", align="start",
      title_size=14, sub_size=12)

# data & ops row
A.box(320, 790, 200, 116, "Audit log", ["JSON lines on stdout", "PII masked"], "data")
A.box(540, 790, 200, 116, "Prometheus", ["scrapes /metrics", "latency · status mix", "reason codes"], "data")
A.box(760, 790, 220, 116, "File volumes", ["reference_store — ref images", "run_store — run photos",
                                           "evaluations — sets · reports"], "data", sub_size=12)
A.box(1000, 790, 200, 116, "registry-sync job", ["one-shot on start-up", "validates + applies diffs",
                                                 "tools/sync_registry.py"], "ci", sub_size=12)
A.box(1240, 790, 200, 116, "registry/  (optional)", ["cards/<cc>/*.yaml", "refs/<family>/<variant>/",
                                                     "Git import source"], "data", sub_size=12)

# change pipeline strip (outside the boundary)
A.box(290, 976, 1180, 132, "Change pipeline", [
    "Day to day: web console → PUT /v1/cards/{id} + reference uploads → stored in Qdrant → live on every worker within 5 s",
    "From Git (optional): YAML + reference images → pull request → CI tools/evaluate.py (gate: ACCEPT precision ≥ 99 %) → registry-sync / POST /v1/registry/sync",
    "Tuning: PATCH /v1/settings (overrides kept in Qdrant)  ·  regression check: POST /v1/evaluations  ·  rollback: re-sync previous revision or restore a Qdrant snapshot"],
      "ci", title_size=14.5, sub_size=12)

# arrows
A.arrow([(240, 318), (318, 318)], "1–2 photos", 279, 304)
A.arrow([(320, 344), (242, 344)], "JSON", 281, 358)
A.arrow([(240, 470), (318, 470)], "admin API", 279, 456)
A.arrow([(415, 520), (415, 660), (242, 660)], "result", 330, 646)
A.arrow([(510, 400), (530, 400), (530, 228), (573, 228)])
A.arrow([(830, 284), (830, 298)])
A.arrow([(830, 514), (830, 530)])
A.arrow([(1085, 400), (1148, 400)], "inference", 1117, 386)
A.arrow([(1148, 640), (1087, 640)], "snapshot", 1117, 626)
A.arrow([(1110, 700), (1148, 700)], "writes", 1129, 714)
A.arrow([(560, 740), (560, 764), (420, 764), (420, 788)])
A.arrow([(690, 740), (690, 788)])
A.arrow([(870, 740), (870, 788)])
A.arrow([(1238, 848), (1202, 848)])
A.arrow([(1170, 790), (1170, 742)], "import", 1180, 766, anchor="start")
A.arrow([(1340, 976), (1340, 908)])

# legend
lx, ly = 40, 760
A.text(lx, ly, "Legend", 13, 700, "#2C2C2A", "start")
for i, (st, label) in enumerate([("svc", "Agent service"), ("gpu", "Model runtime (GPU)"), ("data", "Data / ops"),
                                  ("review", "Admin / human step"), ("ci", "Change pipeline"),
                                  ("neutral", "External system")]):
    fill, stroke, _ = STYLES[st]
    y = ly + 22 + i * 26
    A.add(f'<rect x="{lx}" y="{y - 8}" width="16" height="16" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    A.text(lx + 26, y, label, 12.5, 400, "#44443F", "start")
A.save("architecture.svg", 1130)

# =====================================================================================
# 2) PROCESS FLOW   (matches CardAgent.process / _decide)
# =====================================================================================
F = Svg(1180, 3000, "Card Validation Agent — process flow",
        "Per-request flow from one or two card photos to ACCEPT, REVIEW or REJECT with early exits.")
F.text(40, 30, "Card Validation Agent — request process flow", 22, 700, "#1F1F1D", "start")
F.text(40, 56, "1–2 photos of one card in (front / back) → one JSON decision out. Exits on the right.",
       13.5, 400, "#6B6B66", "start")

MX, MW = 300, 340
CX = MX + MW / 2
EX, EW = 760, 380
GAP = 30
cur = [90]


def step(title, sub, h=64, style="proc"):
    y = cur[0]
    F.box(MX, y, MW, h, title, sub if isinstance(sub, list) else [sub] if sub else [], style)
    cur[0] = y + h + GAP
    F.arrow([(CX, y + h), (CX, cur[0] - 2)])
    return y


def exit_box(y_center, title, sub, style):
    F.box(EX, y_center - 36, EW, 72, title, [sub] if sub else [], style)


def side_exit(y_box, label, title, sub, style, dashed=False, h=64):
    yc = y_box + h / 2
    F.arrow([(MX + MW, yc), (EX - 2, yc)], label, 700, yc - 14, dashed=dashed)
    exit_box(yc, title, sub, style)


def decision(lines, label, title, sub, style, hw=180, hh=54, cont="no"):
    y = cur[0]
    cy = y + hh
    F.diamond(CX, cy, hw, hh, lines)
    F.arrow([(CX + hw, cy), (EX - 2, cy)], label, 700, cy - 14)
    exit_box(cy, title, sub, style)
    F.text(CX + 14, y + 2 * hh + 14, cont, 12, 500, "#55554F", "start")
    cur[0] = y + 2 * hh + GAP
    F.arrow([(CX, y + 2 * hh), (CX, cur[0] - 2)])


def section(y1, y2, label, sub):
    F.add(f'<line x1="250" y1="{y1}" x2="250" y2="{y2}" stroke="#C9C9C3" stroke-width="2"/>')
    F.text(234, y1 + 14, label, 14, 700, "#44443F", "end")
    F.text(234, y1 + 34, sub, 12, 400, "#7A7A74", "end")


# --- intake
s0 = cur[0]
F.pill(MX, cur[0], MW, 56, "Photos + request_id received", "neutral", "fields image | front | back | images")
cur[0] += 56 + GAP
F.arrow([(CX, cur[0] - GAP), (CX, cur[0] - 2)])
y = step("Upload checks", "jpeg / png / webp · ≤ 8 MB each · ≤ 2 photos")
side_exit(y, "invalid", "HTTP 415 / 413 / 422", "rejected before processing", "reject")
decision(["same request_id + same photos?", "idempotency cache"], "yes", "Return cached result",
         "same answer for capture-system retries", "neutral")
y = step("1 · Quality gate  (each photo)", "resolution · normalised sharpness · exposure · glare")
decision(["At least one photo usable?"], "no", "REJECT · RETAKE_REQUIRED",
         "+ issue codes (or INVALID_IMAGE if not decodable)", "reject", cont="yes")
section(s0, cur[0] - GAP, "Intake", "fail fast, cheap checks")

# --- identify
s1 = cur[0]
step("2 · Detect & straighten card  (each photo)", "colour segmentation → edges · warp to 1012×638")
step("3 · Full-card OCR  (each photo)", "PaddleOCR · 0° and 180° · text pooled across photos")
step("4 · Query features  (each photo)", "DINOv2 vector · ORB keypoints · Lab body colour")
step("5 · Stage 1 — shortlist (recall)", "Qdrant top-20 references ∪ keyword inverted index")
step("6 · Stage 2 — verify (precision)", "ORB + RANSAC on ≤ 24 refs (front and back) → inliers")
step("7 · Family scoring", "text · embedding · geometry · colour · fused over photos")
decision(["Two photos match", "two different cards?"], "yes", "REVIEW · SIDES_MISMATCH",
         "each photo strongly matches another family", "review")
decision(["score ≥ min AND", "margin over runner-up ≥ δ?"], "no", "REVIEW · AMBIGUOUS_CARD_TYPE",
         "or REJECT · UNKNOWN_CARD_TYPE if score < 0.30", "review", hw=190, hh=56, cont="yes")
y = step("8 · Variant scoring (inside family)", "colour (front only) · keyword · geometry · prefix")
side_exit(y, "tie", "flag AMBIGUOUS_VARIANT", "continue — decision becomes REVIEW", "review", dashed=True)
y = step("9 · Assign photos to sides", "match to front / back references + declared side")
side_exit(y, "side missing", "note missing_sides", "continue — that side's fields are not read", "neutral",
          dashed=True)
section(s1, cur[0] - GAP, "Identify", "which card, tier and side?")

# --- read & validate
s2 = cur[0]
step("10 · Align each side to its reference", "RANSAC homography if inliers ≥ 25")
step("11 · Extract fields from their side", "region OCR → full-text regex → local VLM*")
step("12 · Rules engine (card definition)", ["regex · length · expiry · date order",
                                             "Emirates ID · Luhn · prefix · lookup"], h=80)
section(s2, cur[0] - GAP, "Read & validate", "fields + business rules")

# --- decide
s3 = cur[0]
decision(["Required side missing?"], "yes", "REJECT · SIDE_REQUIRED:<side>",
         "+ RETAKE_REQUIRED if that side's photo was unusable", "reject")
decision(["Card expired?"], "yes", "REJECT · CARD_EXPIRED", "valid_until is in the past", "reject")
decision(["Any open issue?", "unreadable · low confidence", "rule failed · ambiguous variant · VLM"], "yes",
         "REVIEW + reason codes", "RULE_FAILED:<rule>:<field>, LOW_CONFIDENCE …", "review", hw=200, hh=60)
F.pill(MX, cur[0], MW, 56, "ACCEPT", "accept", "card, variant, fields, photos and checks returned")
end = cur[0] + 56
section(s3, end, "Decide", "status + reason codes")

fy = end + 40
F.text(40, fy, "Every decision → JSON (schema 2.1) · run stored in Qdrant + photos in run_store (review queue, console)"
               " · PII-masked audit log · Prometheus metrics", 12.5, 500, "#44443F", "start")
F.text(40, fy + 22, "POST /v1/cards/validate/stream runs the same flow and emits each step as a server-sent event.",
       12.5, 400, "#44443F", "start")
F.text(40, fy + 44, "* VLM is optional, self-hosted (Ollama), off by default; its output is re-checked by the same regex"
                    " and always triggers REVIEW.", 12, 400, "#7A7A74", "start", italic=True)
F.save("process-flow.svg", int(fy + 70))
print("wrote", OUT)
