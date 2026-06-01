import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Smart CV Analyzer")

uploaded_file = st.file_uploader("Upload Your CV", type=["pdf"])

degree = st.selectbox(
    "Select Your Degree",
    [
        "BS Computer Science",
        "Software Engineering",
        "Data Science",
        "BBA",
        "MBA",
        "Civil Engineering"
    ]
)

job_role = st.text_input("Target Job Role")

job_desc = st.text_area("Paste Job Description")

def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

if uploaded_file and job_desc:
    cv_text = extract_text(uploaded_file)

    docs = [cv_text, job_desc]

    vectors = CountVectorizer().fit_transform(docs)
    score = cosine_similarity(vectors)[0][1]

    ats_score = round(score * 100, 2)

    st.subheader("ATS Score")
    st.success(f"{ats_score}%")

    jd_words = set(job_desc.lower().split())
    cv_words = set(cv_text.lower().split())

    missing = jd_words - cv_words

    st.subheader("Missing Keywords")
    st.write(list(missing)[:20])

    st.subheader("Suggested Skills")

    if degree == "BS Computer Science":
        st.write("Python, SQL, Data Structures, OOP, Git")

    elif degree == "Software Engineering":
        st.write("Java, Python, Software Design, Testing, Git")

    elif degree == "Data Science":
        st.write("Python, Machine Learning, SQL, Power BI")

    elif degree == "BBA":
        st.write("Marketing, Communication, Excel")

    elif degree == "MBA":
        st.write("Leadership, Management, Business Strategy")

    elif degree == "Civil Engineering":
        st.write("AutoCAD, Project Management, Surveying")