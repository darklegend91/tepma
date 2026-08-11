"""Render a profile dict (see profile_schema.py) into a clean single-column resume PDF."""
from fpdf import FPDF

MARGIN = 18
ACCENT = (40, 60, 110)
RULE = (180, 185, 200)
BODY = (35, 35, 35)
MUTED = (110, 110, 110)


def _latin(s: str) -> str:
    return (s or "").encode("latin-1", "replace").decode("latin-1")


class ResumePDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(True, margin=MARGIN)
        self.add_page()

    def section(self, title: str):
        self.ln(3)
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*ACCENT)
        self.cell(0, 6, _latin(title.upper()), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.set_line_width(0.3)
        y = self.get_y()
        self.line(MARGIN, y, 210 - MARGIN, y)
        self.ln(2)
        self.set_text_color(*BODY)

    def entry_header(self, left: str, right: str):
        self.set_font("helvetica", "B", 10.5)
        right_w = 40
        self.cell(210 - 2 * MARGIN - right_w, 5.5, _latin(left))
        self.set_font("helvetica", "", 9.5)
        self.set_text_color(*MUTED)
        self.cell(right_w, 5.5, _latin(right), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*BODY)

    def sub(self, text: str):
        self.set_font("helvetica", "I", 9.5)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*BODY)

    def body(self, text: str):
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")

    def bullet(self, text: str):
        self.set_font("helvetica", "", 10)
        self.cell(5, 5, "-")
        self.multi_cell(210 - 2 * MARGIN - 5, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")


def render_resume(profile: dict) -> bytes:
    pdf = ResumePDF()

    # Header: name, role, contact line
    pdf.set_font("helvetica", "B", 22)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, _latin(profile.get("name", "")), new_x="LMARGIN", new_y="NEXT")
    if profile.get("target_role"):
        pdf.set_font("helvetica", "", 12)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, _latin(profile["target_role"]), new_x="LMARGIN", new_y="NEXT")
    contact = "  |  ".join(
        v for v in (profile.get("email"), profile.get("phone"), profile.get("location")) if v
    )
    if contact:
        pdf.set_font("helvetica", "", 9.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, _latin(contact), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*BODY)

    if profile.get("summary"):
        pdf.section("Summary")
        pdf.body(profile["summary"])

    if profile.get("experience"):
        pdf.section("Experience")
        for job in profile["experience"]:
            dates = " - ".join(v for v in (job.get("start"), job.get("end")) if v)
            pdf.entry_header(f"{job.get('title', '')} - {job.get('company', '')}", dates)
            for b in job.get("bullets", []):
                pdf.bullet(b)
            pdf.ln(1.5)

    if profile.get("projects"):
        pdf.section("Projects")
        for proj in profile["projects"]:
            pdf.entry_header(proj.get("name", ""), "")
            if proj.get("technologies"):
                pdf.sub(proj["technologies"])
            pdf.body(proj.get("description", ""))
            pdf.ln(1.5)

    if profile.get("education"):
        pdf.section("Education")
        for edu in profile["education"]:
            pdf.entry_header(f"{edu.get('degree', '')} - {edu.get('institution', '')}", edu.get("year", ""))
            if edu.get("details"):
                pdf.sub(edu["details"])
            pdf.ln(1)

    if profile.get("skills"):
        pdf.section("Skills")
        pdf.body(", ".join(profile["skills"]))

    if profile.get("achievements"):
        pdf.section("Achievements")
        for a in profile["achievements"]:
            pdf.bullet(a)

    return bytes(pdf.output())
