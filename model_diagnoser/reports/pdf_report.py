from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf_report(results, output_file="model_diagnosis_report.pdf"):
    """
    Generate a PDF report for model diagnostics.
    """

    c = canvas.Canvas(output_file, pagesize=letter)

    y = 750

    c.setFont("Helvetica", 12)

    c.drawString(200, y, "Model Diagnosis Report")

    y -= 40

    for key, value in results.items():

        c.drawString(50, y, f"{key}:")

        y -= 20

        text = str(value)

        for line in text.split("\n"):

            c.drawString(70, y, line)

            y -= 15

            if y < 50:
                c.showPage()
                y = 750

        y -= 20

    c.save()

    print(f"PDF report generated: {output_file}")