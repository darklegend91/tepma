"""Render a profile dict (see profile_schema.py) into a clean single-column resume PDF."""
import json
import re
from pathlib import Path

from fpdf import FPDF

MARGIN = 18
ACCENT = (40, 60, 110)
RULE = (180, 185, 200)
BODY = (35, 35, 35)
MUTED = (110, 110, 110)

# The built-in PDF fonts are Latin-1 only. A resume that still contains Devanagari or
# Gurmukhi - a name nobody could transliterate, say - must not be typeset as a row of
# question marks, so those scripts pull in a real Unicode font instead.
FONT_DIR = Path(__file__).parent / "data" / "fonts"
DEVANAGARI = re.compile("[ऀ-ॿ]")
GURMUKHI = re.compile("[਀-੿]")
# Candidates per script, best first. Plain TTFs are embedded straight from the system
# path; a .ttc is a collection fpdf2 cannot open, so face 0 is extracted into FONT_DIR
# once. Arial Unicode is listed first because it is the only one of these that carries
# Latin, Devanagari and Gurmukhi together - the Devanagari-only faces would typeset the
# section headings ("SUMMARY", "EXPERIENCE") as blanks.
SYSTEM_FONTS = {
    "deva": ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
             "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"),
    "guru": ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
             "/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc"),
}


def _unicode_font_file(script: str) -> Path | None:
    """A TTF covering `script` plus Latin, or None if this machine has no such font."""
    for source in SYSTEM_FONTS[script]:
        source_path = Path(source)
        if not source_path.is_file():
            continue
        if source_path.suffix.lower() != ".ttc":
            return source_path
        target = FONT_DIR / (source_path.stem + ".ttf")
        if target.is_file():
            return target
        try:
            from fontTools.ttLib import TTFont
            FONT_DIR.mkdir(parents=True, exist_ok=True)
            TTFont(str(source_path), fontNumber=0).save(str(target))
        except Exception:
            continue
        if target.is_file():
            return target
    return None


def _script_of(profile: dict) -> str | None:
    """Which Indic script, if any, the resume text needs.

    Bookkeeping keys are skipped: _corrections records what a value used to be, so an
    entirely English resume would otherwise drag in a Unicode font for text nobody prints.
    """
    text = json.dumps({k: v for k, v in profile.items() if not k.startswith("_")},
                      ensure_ascii=False)
    if DEVANAGARI.search(text):
        return "deva"
    if GURMUKHI.search(text):
        return "guru"
    return None


class ResumePDF(FPDF):

    def __init__(self, script: str | None = None):
        super().__init__("P", "mm", "A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(True, margin=MARGIN)
        # "helvetica" unless the resume needs an Indic script, in which case every style
        # maps to the one embedded face - these fonts have no separate italic, and a
        # missing style is what makes fpdf2 fall back to Latin-1 and print "?".
        self.family = "helvetica"
        if script:
            font_file = _unicode_font_file(script)
            if font_file:
                for style in ("", "B", "I"):
                    self.add_font("resume", style, str(font_file))
                self.family = "resume"
                try:
                    self.set_text_shaping(True)   # needs uharfbuzz; matras are misplaced without it
                except Exception:
                    pass
        self.add_page()

    def _text(self, s: str) -> str:
        """Latin-1 is lossy, so it is only ever applied to the built-in fonts."""
        if self.family != "helvetica":
            return s or ""
        return (s or "").encode("latin-1", "replace").decode("latin-1")

    def section(self, title: str):
        self.ln(3)
        self.set_font(self.family, "B", 11)
        self.set_text_color(*ACCENT)
        self.cell(0, 6, self._text(title.upper()), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.set_line_width(0.3)
        y = self.get_y()
        self.line(MARGIN, y, 210 - MARGIN, y)
        self.ln(2)
        self.set_text_color(*BODY)

    def entry_header(self, left: str, right: str):
        self.set_font(self.family, "B", 10.5)
        right_w = 40
        self.cell(210 - 2 * MARGIN - right_w, 5.5, self._text(left))
        self.set_font(self.family, "", 9.5)
        self.set_text_color(*MUTED)
        self.cell(right_w, 5.5, self._text(right), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*BODY)

    def sub(self, text: str):
        self.set_font(self.family, "I", 9.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 5, self._text(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*BODY)

    def body(self, text: str):
        self.set_font(self.family, "", 10)
        self.multi_cell(0, 5, self._text(text), new_x="LMARGIN", new_y="NEXT")

    def bullet(self, text: str):
        self.set_font(self.family, "", 10)
        self.cell(5, 5, "-")
        self.multi_cell(210 - 2 * MARGIN - 5, 5, self._text(text), new_x="LMARGIN", new_y="NEXT")


def render_resume(profile: dict) -> bytes:
    pdf = ResumePDF(_script_of(profile))

    # Header: name, role, contact line
    pdf.set_font(pdf.family, "B", 22)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, pdf._text(profile.get("name", "")), new_x="LMARGIN", new_y="NEXT")
    if profile.get("target_role"):
        pdf.set_font(pdf.family, "", 12)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, pdf._text(profile["target_role"]), new_x="LMARGIN", new_y="NEXT")
    contact = "  |  ".join(
        v for v in (profile.get("email"), profile.get("phone"), profile.get("location")) if v
    )
    if contact:
        pdf.set_font(pdf.family, "", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, pdf._text(contact), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BODY)

    # Write Summary
    if profile.get("summary"):
        pdf.section("Summary")
        pdf.body(profile["summary"])

    # Write Experience
    if profile.get("experience"):
        pdf.section("Experience")
        for job in profile["experience"]:
            dates = " - ".join(v for v in (job.get("start"), job.get("end")) if v)
            pdf.entry_header(f"{job.get('title', '')} - {job.get('company', '')}", dates)
            for b in job.get("bullets", []):
                pdf.bullet(b)
            pdf.ln(1.5)
    
    # Write Projects
    if profile.get("projects"):
        pdf.section("Projects")
        for proj in profile["projects"]:
            pdf.entry_header(proj.get("name", ""), "")
            if proj.get("technologies"):
                pdf.sub(proj["technologies"])
            pdf.body(proj.get("description", ""))
            pdf.ln(1.5)
            
    # Write Education
    if profile.get("education"):
        pdf.section("Education")
        for edu in profile["education"]:
            where = ", ".join(part for part in (edu.get("institution", ""),
                                                edu.get("location", "")) if part)
            pdf.entry_header(f"{edu.get('degree', '')} - {where}", edu.get("year", ""))
            if edu.get("details"):
                pdf.sub(edu["details"])
            pdf.ln(1)
    
    # Write Skills
    if profile.get("skills"):
        pdf.section("Skills")
        pdf.body(", ".join(profile["skills"]))

    # Write Achievements
    if profile.get("achievements"):
        pdf.section("Achievements")
        for a in profile["achievements"]:
            pdf.bullet(a)

    return bytes(pdf.output())