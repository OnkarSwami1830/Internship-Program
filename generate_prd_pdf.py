from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors


CONTENT = """
Project Requirement Document
DAY 1

AI + NLP Project

AI Job Description Skill Extraction System
Automatically identify the job role and required skills from a Job Description using NLP.

1. Problem Statement
Recruiters receive hundreds of Job Descriptions (JDs), but manually identifying the job role and required skills is time-consuming and inconsistent. The goal is to build an AI system that automatically extracts structured information from unstructured JD text.

2. Business Objective
Create a smart recruitment solution that helps HR teams:

- Reduce manual screening time
- Standardize skill extraction
- Improve candidate-job matching
- Generate structured recruitment data

3. Technical Objective
Develop an NLP pipeline that:

- Accepts a JD as input
- Detects the job role
- Extracts technical and soft skills
- Returns JSON output
- Can be deployed using Flask or FastAPI
The required technology stack in the task includes Python, Pandas, NumPy, NLTK, spaCy, Scikit-learn, Transformers, Matplotlib, Seaborn, Power BI/Tableau, and Flask/FastAPI.

4. Understanding the NLP Concepts

Concept | Meaning
Keyword Extraction | Finds important words like Python, SQL
Entity Extraction | Identifies named entities such as skills, tools, roles
Classification | Predicts the job category (Data Scientist, Java Developer)
Information Extraction | Extracts structured fields like role, skills, experience

These are the core concepts listed in the Day 1 task.

5. Project Input
The input is a plain-text Job Description.

Example
We require a Data Scientist with Python, Pandas, NumPy, Scikit-learn, SQL and AWS experience.

6. Expected Output
JSON

{
  "role": "Data Scientist",
  "skills": [
    "Python",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "SQL",
    "AWS"
  ]
}

This output format matches the example in the PDF.

7. Technologies
Python - NLP development
Jupyter Notebook - Experimentation
Pandas & NumPy - Data processing
NLTK & spaCy - Text preprocessing
Transformers - Advanced NLP
Power BI / Tableau - Analytics dashboard

8. Success Criteria
The project is successful if it can:

- Correctly identify the job role
- Extract 90%+ of required skills from a JD
- Produce structured JSON output
- Process JDs in real time through an API

9. Future Scope (Resume-worthy)
You can improve this into a production-level AI project by adding:

1. Skill recommendations for missing skills
2. Experience extraction (3+ years, 5 years)
3. Education extraction
4. Salary & location extraction
5. Resume vs JD matching score
6. AI suggestions to improve the Job Description

Deliverable
File name: Project_Requirement_Document.pdf

This is a strong Day 1 document that aligns with the internship task and is suitable for submission.
"""


def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        alignment=TA_CENTER,
        fontSize=18,
        leading=24,
        spaceAfter=18,
        textColor=colors.HexColor('#1F2A44'),
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=13,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.HexColor('#283B5B'),
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontSize=10.5,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=body_style,
        leftIndent=18,
        firstLineIndent=-10,
        bulletIndent=10,
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=body_style,
        fontName='Courier',
        backColor=colors.HexColor('#F4F6F9'),
        borderColor=colors.HexColor('#D7DFEA'),
        borderWidth=1,
        borderPadding=6,
        spaceAfter=10,
    )

    story = []
    lines = CONTENT.strip().splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            story.append(Spacer(1, 6))
            i += 1
            continue

        if line.startswith('Project Requirement Document'):
            story.append(Paragraph(line, title_style))
            i += 1
            continue

        if line.startswith('DAY 1'):
            story.append(Paragraph(line, body_style))
            i += 1
            continue

        if line.startswith('AI + NLP Project'):
            story.append(Paragraph(line, heading_style))
            i += 1
            continue

        if line.startswith('AI Job Description Skill Extraction System'):
            story.append(Paragraph(line, heading_style))
            i += 1
            continue

        if line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.') or line.startswith('6.') or line.startswith('7.') or line.startswith('8.') or line.startswith('9.'):
            story.append(Paragraph(line, heading_style))
            i += 1
            continue

        if line.startswith('- '):
            bullet_items = []
            while i < len(lines) and lines[i].startswith('- '):
                bullet_items.append(lines[i][2:].strip())
                i += 1
            story.append(ListFlowable([ListItem(Paragraph(item, bullet_style), value='circle') for item in bullet_items], bulletType='bullet', bulletColor=colors.HexColor('#1F2A44')))
            continue

        if line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. ') or line.startswith('4. ') or line.startswith('5. ') or line.startswith('6. '):
            story.append(Paragraph(line, body_style))
            i += 1
            continue

        if line.startswith('{') or line.startswith('"role"') or line.startswith('"skills"') or line.startswith('  "role"') or line.startswith('  "skills"') or line.startswith('    "Python"') or line.startswith('    "Pandas"') or line.startswith('    "NumPy"') or line.startswith('    "Scikit-learn"') or line.startswith('    "SQL"') or line.startswith('    "AWS"') or line.startswith('}'):
            story.append(Paragraph(line, code_style))
            i += 1
            continue

        if line.startswith('Concept') or line.startswith('Keyword Extraction') or line.startswith('Entity Extraction') or line.startswith('Classification') or line.startswith('Information Extraction'):
            story.append(Paragraph(line, body_style))
            i += 1
            continue

        story.append(Paragraph(line, body_style))
        i += 1

    doc.build(story)


if __name__ == '__main__':
    output_path = r"C:\Users\Onkar Swami\Desktop\TECNO\ai_job\Project_Requirement_Document.pdf"
    build_pdf(output_path)
    print(f"PDF generated: {output_path}")
