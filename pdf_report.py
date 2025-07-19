from fpdf import FPDF
import datetime
import os

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, "Meeting Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_transcript(self, transcript):
        self.set_font("DejaVu", size=12)
        self.cell(0, 10, "Transcript:", ln=True)
        self.ln(5)
        for entry in transcript:
            timestamp = entry.get("timestamp", "")
            text = entry.get("text", "")
            self.multi_cell(0, 10, f"[{timestamp}] {text}")
            self.ln(1)

    def add_summary(self, insight):
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 10, "Summary / Insight", ln=True)
        self.set_font("DejaVu", size=12)
        self.multi_cell(0, 10, insight)


def generate_pdf_report(transcript, insight):
    pdf = PDF()
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.add_font("DejaVu", "B", FONT_PATH, uni=True)
    pdf.add_font("DejaVu", "I", FONT_PATH, uni=True)

    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.add_transcript(transcript)
    pdf.ln(10)
    pdf.add_summary(insight)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"meeting_report_{now}.pdf"
    path = os.path.join("pdf_reports", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf.output(path)
    return path
