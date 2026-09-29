skills = [
    "python",
    "sql",
    "power bi",
    "tableau",
    "excel",
    "aws",
    "java",
    "docker",
    "react",
    "spring boot"
]


def extract_skills(text):
    found = []

    for skill in skills:
        if skill in text.lower():
            found.append(skill.title())

    return found


job = """
We require a Data Scientist with Python, SQL,
Power BI and AWS experience.
"""

print(extract_skills(job))

# Example usage for multiple jobs
jobs = {
    "Data Scientist": "Python, SQL, AWS",
    "Data Analyst": "Excel, Power BI, Tableau",
    "Java Developer": "Java, Spring Boot",
    "Full Stack Developer": "React, Java, SQL",
}

for role, jd in jobs.items():
    print(role, "->", extract_skills(jd))
