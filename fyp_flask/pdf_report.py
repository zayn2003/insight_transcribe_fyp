from fpdf import FPDF
import datetime
import os

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Meeting Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
    
    def add_transcript(self, transcript):
        self.set_font("Arial", size=12)
        self.cell(0, 10, "Transcript:", ln=True)
        self.ln(5)
        
        for line in transcript:
            if line.startswith("["):
                self.multi_cell(0, 6, line)
                self.ln(3)

    def add_summary(self, insight):
        # Remove prefixes
        if insight.startswith("FINAL INSIGHT:"):
            insight = insight.replace("FINAL INSIGHT:", "")
        elif insight.startswith("INSIGHT:"):
            insight = insight.replace("INSIGHT:", "")
        
        # Clean any remaining special characters
        insight = insight.replace('•', '-')
        
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Summary / Insight", ln=True)
        self.set_font("Arial", size=12)
        self.multi_cell(0, 6, insight)

def generate_pdf_report(transcript, insight):
    pdf = PDF()
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