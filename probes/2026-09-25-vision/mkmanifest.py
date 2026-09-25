"""Vision-quiz manifest: human-verified game-perception questions for Qwen3.5-4B.

Every label was verified by a human against zoomed crops; programmatic
analysis only proposed candidates. Run: python mkquiz_manifest.py -> writes
vision-manifest.json (image keys resolved by quiz.py on the bench host).
"""
import json

Q = []
qid = [0]


def add(cat, images, question, options, answer, pair=None, transform=None, control=False):
    qid[0] += 1
    assert answer in [o[0] for o in options] or options == ["OCR"], (qid[0], answer)
    Q.append({
        "id": f"q{qid[0]:03d}", "cat": cat, "images": images,
        "question": question, "options": options, "answer": answer,
        "pair": pair, "transform": transform, "control": control,
    })


def ch(question, options):
    return question + " " + " ".join(f"{L}) {t}" for L, t in options) + \
        " Answer with exactly one letter."


M3 = "cap:menu3"
M2 = "cap:menu2"
M1 = "cap:menu1"
PL = "cap:play"
DR = "cap:drplay"
FX = "fix:"

# ---- A1. game-select quadrant (3-way) ----
GSEL = {"T": "TETRIS", "D": "Dr. MARIO", "M": "MIXED MATCH"}
gsel_q = [("A", "TETRIS"), ("B", "Dr. MARIO"), ("C", "MIXED MATCH")]
for img, lab in [(f"{M2}/10-gsel-u3-up", "A"), (f"{M2}/14-gsel-tetris", "A"),
                 (f"{M1}/03-nav3-up", "A"),
                 (f"{M2}/07-gsel-r1-right", "B"), (f"{M3}/24-gsel-drmario", "B"),
                 (f"{M1}/03-nav5-right", "B"),
                 (f"{M2}/03-gsel-d1-down", "C"), (f"{M1}/03-nav0-down", "C")]:
    add("menu-gsel", [img], ch("Which game quadrant is highlighted?", gsel_q), gsel_q, lab)

# ---- A2. tetris submenu row (3-way) ----
TSUB = [("A", "1PLAYER GAME"), ("B", "2PLAYER GAME"), ("C", "VS. COM")]
for img, lab in [(f"{M3}/02-tsub-row1", "A"), (f"{M1}/04-after-A", "A"),
                 (f"{M3}/03-tsub-row2", "B"), (f"{M1}/05-nav0-down", "B"),
                 (f"{M3}/04-tsub-row3", "C"), (f"{M1}/05-nav1-down", "C")]:
    add("menu-tsub", [img], ch("Which row is the cursor on?", TSUB), TSUB, lab)

# ---- A3. dr.mario submenu row (3-way) ----
for img, lab in [(f"{M2}/17-drm-d1-down", "A"), (f"{M3}/25-dsub-row1", "A"),
                 (f"{M2}/18-drm-d2-down", "B"), (f"{M2}/20-drm-u1-up", "B"),
                 (f"{M2}/21-drm-l1-left", "B"), (f"{M2}/19-drm-d3-down", "C")]:
    add("menu-dsub", [img], ch("Which row is the heart cursor on?", TSUB), TSUB, lab)

# ---- A4. tetris music highlight (4-way) ----
MUS = [("A", "MUSIC-1"), ("B", "MUSIC-2"), ("C", "MUSIC-3"), ("D", "OFF")]
for img, lab in [(f"{M3}/06-topt-0", "A"),
                 (f"{M3}/07-topt-0-down", "B"), (f"{M3}/11-topt-4-up", "B"),
                 (f"{M3}/08-topt-1-down", "C"), (f"{M3}/12-topt-5-right", "C"),
                 (f"{M3}/09-topt-2-down", "D"), (f"{M3}/10-topt-3-down", "D")]:
    add("menu-music", [img], ch("Which MUSIC item is highlighted in white?", MUS), MUS, lab)

# ---- A5/B. dr.mario options: virus OCR / speed / music ----
for img, val in [(f"{M3}/26-dopt-0", "00"), (f"{M3}/27-dopt-0-right", "02"),
                 (f"{M3}/28-dopt-1-right", "04"), (f"{M3}/36-dopt-9-left", "02")]:
    add("ocr-virus", [img], "What 2-digit VIRUS LEVEL number is shown? Answer with exactly the digits.",
        ["OCR"], val)
SPD = [("A", "LOW"), ("B", "MED"), ("C", "HI")]
for img, lab in [(f"{M3}/26-dopt-0", "B"), (f"{M3}/29-dopt-2-down", "B"),
                 (f"{M3}/30-dopt-3-right", "C"), (f"{M3}/34-dopt-7-up", "C")]:
    add("menu-speed", [img], ch("Which SPEED is selected (triangle marker)?", SPD), SPD, lab)
DMUS = [("A", "FEVER"), ("B", "CHILL"), ("C", "OFF")]
for img, lab in [(f"{M3}/26-dopt-0", "A"), (f"{M3}/31-dopt-4-down", "A"),
                 (f"{M3}/32-dopt-5-right", "B"),
                 (f"{M3}/33-dopt-6-right", "C"), (f"{M3}/35-dopt-8-up", "C")]:
    add("menu-dmusic", [img], ch("Which MUSIC TYPE is selected (green box)?", DMUS), DMUS, lab)

# ---- B. OCR exact ----
HS = f"{M3}/15-tgame-0"
add("ocr-hiscore", [HS], "In the high-score table, what 6-digit SCORE is shown for OTASAN? Answer with exactly the digits.", ["OCR"], "007500")
add("ocr-hiscore", [HS], "In the high-score table, what LV is shown for HOWARD? Answer with exactly the digits.", ["OCR"], "09")
add("ocr-hiscore", [HS], "In the high-score table, what NAME is ranked 3rd? Answer with exactly the name.", ["OCR"], "LANCE")
add("ocr-hiscore", [HS], "In the high-score table, what 6-digit SCORE is shown for rank 1? Answer with exactly the digits.", ["OCR"], "010000")
for img, val in [(f"{PL}/f0000", "000000"), (f"{PL}/f0060", "000003"),
                 (f"{FX}gameplay-68.png", "000068"), (f"{FX}gameplay-99.png", "000099")]:
    add("ocr-score", [img], "What 6-digit SCORE is shown? Answer with exactly the digits.", ["OCR"], val)
add("ocr-stats", [f"{FX}gameplay-99.png"], "In STATISTICS, top to bottom the rows are T, S, Z, square, L, J, I. What count is shown for the L row (5th)? Answer with exactly the digits.", ["OCR"], "003")
add("ocr-stats", [f"{FX}gameplay-99.png"], "In STATISTICS, what count is shown for the J row (6th)? Answer with exactly the digits.", ["OCR"], "002")
add("ocr-stats", [f"{FX}gameplay-68.png"], "In STATISTICS, what count is shown for the J row (6th)? Answer with exactly the digits.", ["OCR"], "002")
add("ocr-stats", [f"{FX}gameplay-68.png"], "In STATISTICS, what count is shown for the T row (1st)? Answer with exactly the digits.", ["OCR"], "001")
add("ocr-drm", [f"{M3}/40-dgame-0"], "What LEVEL number is shown on the right panel? Answer with exactly the digits.", ["OCR"], "02")
add("ocr-drm", [f"{M3}/40-dgame-0"], "What VIRUS number is shown on the right panel? Answer with exactly the digits.", ["OCR"], "12")
add("ocr-drm", [f"{DR}/g000"], "What LEVEL number is shown on the right panel? Answer with exactly the digits.", ["OCR"], "06")
add("ocr-drm", [f"{DR}/g000"], "What VIRUS number is shown on the right panel? Answer with exactly the digits.", ["OCR"], "26")
add("ocr-drm", [f"{M3}/40-dgame-0"], "What SPEED word is shown on the right panel? Answer with exactly the word.", ["OCR"], "HI")
add("ocr-drm", [f"{DR}/g000"], "What SPEED word is shown on the right panel? Answer with exactly the word.", ["OCR"], "MED")

# ---- C. tetris geometry ----
THIRD = [("A", "left third"), ("B", "center third"), ("C", "right third")]
thirds = [(f"{PL}/f0001", "A"), (f"{PL}/f0011", "A"), (f"{PL}/f0002", "A"),
          (f"{PL}/f0020", "B"), (f"{PL}/f0005", "B"), (f"{PL}/f0027", "B"),
          (f"{PL}/f0049", "C"), (f"{PL}/f0036", "C"), (f"{PL}/f0068", "C")]
for img, lab in thirds:
    add("geo-third", [img], ch("In which horizontal third of the Tetris well is the falling piece?", THIRD), THIRD, lab)
DIR = [("A", "left"), ("B", "right"), ("C", "down")]
dirpairs = [(f"{PL}/f0000", f"{PL}/f0001", "A"), (f"{PL}/f0010", f"{PL}/f0011", "A"),
            (f"{PL}/f0019", f"{PL}/f0020", "A"),
            (f"{PL}/f0004", f"{PL}/f0005", "B"), (f"{PL}/f0025", f"{PL}/f0026", "B"),
            (f"{PL}/f0048", f"{PL}/f0049", "B"),
            (f"{PL}/f0001", f"{PL}/f0002", "C"), (f"{PL}/f0026", f"{PL}/f0027", "C"),
            (f"{PL}/f0051", f"{PL}/f0052", "C")]
for a, b, lab in dirpairs:
    add("geo-dir", [a, b], ch("Two consecutive gameplay frames, first then second. In which direction did the falling piece move between them?", DIR), DIR, lab)
add("geo-col", [f"{PL}/f0084"], ch("The well has 10 columns numbered 0 to 9 from left to right. Which column contains a block in the bottom row?", [("A", "3"), ("B", "1"), ("C", "7")]), [("A", "3"), ("B", "1"), ("C", "7")], "A")
add("geo-col", [f"{PL}/f0024"], ch("The well has 10 columns numbered 0 to 9 from left to right. Which column contains a block in the bottom row?", [("A", "0"), ("B", "6"), ("C", "3")]), [("A", "0"), ("B", "6"), ("C", "3")], "C")
add("geo-col", [f"{FX}gameplay-68.png"], ch("The well has 10 columns numbered 0 to 9 from left to right. Which column contains a block in the bottom row?", [("A", "8"), ("B", "4"), ("C", "2")]), [("A", "8"), ("B", "4"), ("C", "2")], "B")
SIDE = [("A", "left"), ("B", "right")]
for img, lab in [(f"{PL}/f0001", "A"), (f"{PL}/f0011", "A"),
                 (f"{PL}/f0049", "B"), (f"{PL}/f0036", "B")]:
    add("geo-side", [img], ch("Is the falling piece to the LEFT or RIGHT of the settled stack's center?", SIDE), SIDE, lab)

# ---- D. dr.mario geometry ----
PILL3 = [("A", "left third"), ("B", "center third"), ("C", "right third")]
for img, lab in [(f"{DR}/g001", "A"), (f"{DR}/g011", "A"),
                 (f"{DR}/g006", "B"), (f"{M3}/42-dgame-2", "B")]:
    add("drm-third", [img], ch("In which horizontal third of the bottle is the falling pill capsule?", PILL3), PILL3, lab)
PCOL = [("A", "red and yellow"), ("B", "red and blue"), ("C", "red and red"), ("D", "yellow and blue")]
for img, lab in [(f"{DR}/g001", "A"), (f"{DR}/g011", "A"), (f"{DR}/g006", "B"),
                 (f"{M3}/42-dgame-2", "C"), (f"{M3}/44-dgame-4", "C"), (f"{M3}/46-dgame-6", "D")]:
    add("drm-colors", [img], ch("What are the two colors of the falling pill capsule's halves (upper capsule, not landed pieces)?", PCOL), PCOL, lab)
FIRST2 = [("A", "first"), ("B", "second")]
for a, b, lab in [(f"{DR}/g000", f"{M3}/40-dgame-0", "A"), (f"{DR}/g011", f"{M3}/44-dgame-4", "A"),
                  (f"{M3}/40-dgame-0", f"{DR}/g000", "B"), (f"{M3}/44-dgame-4", f"{DR}/g011", "B")]:
    add("drm-more", [a, b], ch("Which image shows MORE virus germs in the bottle, the first or the second?", FIRST2), FIRST2, lab)
BINQ = [("A", "about 10"), ("B", "about 25")]
add("drm-count", [f"{M3}/40-dgame-0"], ch("Roughly how many virus germs are in the bottle?", BINQ), BINQ, "A")
add("drm-count", [f"{DR}/g000"], ch("Roughly how many virus germs are in the bottle?", BINQ), BINQ, "B")

# ---- E. selection ----
add("sel", [f"{FX}gameover-80.png", f"{FX}gameplay-68.png"], ch("Which image shows the GAME OVER text, the first or the second?", FIRST2), FIRST2, "A")
add("sel", [f"{FX}gameplay-99.png", f"{FX}gameover-80.png"], ch("Which image shows the GAME OVER text, the first or the second?", FIRST2), FIRST2, "B")
add("sel", [f"{M2}/01-title", f"{PL}/f0036"], ch("Which image shows the game title screen, the first or the second?", FIRST2), FIRST2, "A")
add("sel", [f"{DR}/g000", f"{M2}/01-title"], ch("Which image shows the game title screen, the first or the second?", FIRST2), FIRST2, "B")
add("sel", [f"{DR}/g000", f"{PL}/f0036"], ch("Which image shows the Dr. Mario bottle, the first or the second?", FIRST2), FIRST2, "A")
add("sel", [f"{PL}/f0048", f"{M3}/44-dgame-4"], ch("Which image shows the Dr. Mario bottle, the first or the second?", FIRST2), FIRST2, "B")

# ---- F. scale probe (2x nearest) ----
add("scale", [f"{M3}/03-tsub-row2"], ch("Which row is the cursor on?", TSUB), TSUB, "B", transform="scale2")
add("scale", [f"{M3}/28-dopt-1-right"], "What 2-digit VIRUS LEVEL number is shown? Answer with exactly the digits.", ["OCR"], "04", transform="scale2")
add("scale", [f"{PL}/f0049"], ch("In which horizontal third of the Tetris well is the falling piece?", THIRD), THIRD, "C", transform="scale2")
add("scale", [f"{DR}/g011"], ch("In which horizontal third of the bottle is the falling pill capsule?", PILL3), PILL3, "A", transform="scale2")
add("scale", [HS], "In the high-score table, what 6-digit SCORE is shown for OTASAN? Answer with exactly the digits.", ["OCR"], "007500", transform="scale2")
add("scale", [f"{M3}/41-dgame-1", f"{M3}/42-dgame-2"], ch("Two consecutive gameplay frames, first then second. In which direction did the falling pill move between them?", DIR), DIR, "C", transform="scale2")

# ---- G. mirror pairs (horizontal flip; answer mirrored) ----
def orig(cat, images):
    for q in Q:
        if q["cat"] == cat and q["images"] == images and q["transform"] is None:
            return q
    raise KeyError((cat, images))

FLIP3 = {"A": "C", "C": "A", "B": "B"}
FLIPD = {"A": "B", "B": "A", "C": "C"}
flips = 0
for img, _ in thirds:
    o = orig("geo-third", [img])
    add("geo-third", [img], o["question"], THIRD, FLIP3[o["answer"]], pair=o["id"], transform="flip")
    flips += 1
for a, b, _ in dirpairs:
    o = orig("geo-dir", [a, b])
    add("geo-dir", [a, b], o["question"], DIR, FLIPD[o["answer"]], pair=o["id"], transform="flip")
    flips += 1
o = orig("geo-col", [f"{PL}/f0084"])
add("geo-col", [f"{PL}/f0084"], ch("The well has 10 columns numbered 0 to 9 from left to right. Which column contains a block in the bottom row?", [("A", "6"), ("B", "8"), ("C", "2")]), [("A", "6"), ("B", "8"), ("C", "2")], "A", pair=o["id"], transform="flip")
o = orig("geo-col", [f"{PL}/f0024"])
add("geo-col", [f"{PL}/f0024"], ch("The well has 10 columns numbered 0 to 9 from left to right. Which column contains a block in the bottom row?", [("A", "9"), ("B", "3"), ("C", "6")]), [("A", "9"), ("B", "3"), ("C", "6")], "C", pair=o["id"], transform="flip")
o = orig("geo-col", [f"{FX}gameplay-68.png"])
add("geo-col", [f"{FX}gameplay-68.png"], ch("The well has 10 columns numbered 0 to 9 from left to right. Which column contains a block in the bottom row?", [("A", "1"), ("B", "5"), ("C", "7")]), [("A", "1"), ("B", "5"), ("C", "7")], "B", pair=o["id"], transform="flip")
flips += 3
for img in [f"{DR}/g001", f"{DR}/g011"]:
    o = orig("drm-third", [img])
    add("drm-third", [img], o["question"], PILL3, FLIP3[o["answer"]], pair=o["id"], transform="flip")
    flips += 1

# ---- H. controls (expect collapse; scored separately) ----
o = orig("geo-dir", [f"{PL}/f0010", f"{PL}/f0011"])
add("control", o["images"], o["question"], DIR, o["answer"], transform="blank", control=True)
o = orig("menu-tsub", [f"{M3}/03-tsub-row2"])
add("control", o["images"], o["question"], TSUB, o["answer"], transform="blank", control=True)
o = orig("geo-dir", [f"{PL}/f0025", f"{PL}/f0026"])
add("control", o["images"], o["question"], DIR, o["answer"], transform="shuffle", control=True)
o = orig("geo-dir", [f"{PL}/f0048", f"{PL}/f0049"])
add("control", o["images"], o["question"], DIR, o["answer"], transform="shuffle", control=True)
add("control", [f"{M3}/28-dopt-1-right"], "What 2-digit VIRUS LEVEL number is shown? Answer with exactly the digits.", ["OCR"], "04", transform="shuffle", control=True)
o = orig("geo-third", [f"{PL}/f0049"])
add("control", o["images"], o["question"], THIRD, o["answer"], transform="shuffle", control=True)

with open("vision-manifest.json", "w") as f:
    json.dump(Q, f, indent=1)
print(f"wrote {len(Q)} questions ({flips} mirror flips, 6 controls)")
