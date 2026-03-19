import streamlit as st
import os
import tempfile
import base64
import re
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# --- 1. Page Configuration & Setup ---
st.set_page_config(page_title="Resume Genie Dashboard", page_icon="🧞", layout="wide", initial_sidebar_state="expanded")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found. Please ensure you have added it to your `.env` file.")
    st.stop()

# --- 2. Global CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Shared Utility Functions ---
@st.cache_resource
def get_llm():
    """Initializes and caches the LLM connection."""
    return ChatGroq(model='llama-3.3-70b-versatile', groq_api_key=groq_api_key)

chat = get_llm()

def extract_text_from_pdf(uploaded_file):
    """Saves uploaded file temporarily, extracts text, and cleans up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        context = "\n\n".join(doc.page_content for doc in documents)
        return context
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def display_pdf(uploaded_file):
    """Embeds a PDF in the Streamlit UI."""
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="750" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def extract_score(response_text):
    """Extracts ATS score using regex."""
    match = re.search(r'\*\*Score:\*\*\s*(\d+)', response_text)
    if match:
        return int(match.group(1))
    return 0

def create_pie_chart(score):
    """Generates the ATS donut chart."""
    df = pd.DataFrame({"Category": ["Match", "Gap"], "Value": [score, 100 - score]})
    fig = px.pie(
        df, values='Value', names='Category', color='Category',
        color_discrete_map={'Match': '#00CC96', 'Gap': '#EF553B'}, hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16)
    fig.update_layout(title_text="Resume Match Analysis", title_x=0.25, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
    return fig

# --- 4. Service Modules ---

def render_resume_evaluator():
    st.markdown('<p class="main-header">📄 AI Resume Evaluator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a resume to get a general score, strengths, weaknesses, and recommended paths.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="eval_uploader")
    
    if uploaded_file and st.button("Evaluate Resume"):
        with st.spinner("Analyzing resume. This may take a few seconds..."):
            try:
                context = extract_text_from_pdf(uploaded_file)
                prompt = PromptTemplate(
                    input_variables=['context', 'question'],
                    template="""
                    You are an advanced resume evaluation assistant. Analyze the provided resume in the context document and score it out of 100 based on clarity, relevance, format, comprehensiveness, and impact.
                    Resume: {context}
                    
                    Structure your response EXACTLY as follows:
                    1. **Score**: [Score out of 100]
                    2. **Strengths**: [List at least 3]
                    3. **Weaknesses**: [List at least 3]
                    4. **Skills Mentioned**: [List found skills]
                    5. **Recommended Skills**: [Suggest additional skills]
                    6. **Next Career Paths**: [Suggest next roles]
                    
                    User Question: {question}
                    """
                )
                formatted_prompt = prompt.format(context=context, question="Please evaluate this resume.")
                
                st.subheader("📊 Evaluation Report")
                def stream_generator():
                    for chunk in chat.stream(formatted_prompt):
                        yield chunk.content
                st.write_stream(stream_generator())
            except Exception as e:
                st.error(f"An error occurred: {e}")

def render_cover_letter_generator():
    st.markdown('<p class="main-header">✉️ AI Cover Letter Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your resume and the target job description to draft a tailored cover letter.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type=["pdf"], key="cl_uploader")
    with col2:
        job_description = st.text_area("2. Paste Job Description", height=200, key="cl_jd")

    if st.button("Generate Cover Letter"):
        if not uploaded_file:
            st.warning("Please upload a resume first.")
        elif not job_description.strip():
            st.warning("Please paste a job description.")
        else:
            with st.spinner("Drafting your personalized cover letter..."):
                try:
                    context = extract_text_from_pdf(uploaded_file)
                    template = f"""
                    Write a professional, compelling cover letter tailored specifically to the job description provided.
                    Emphasize relevant experience matching the role. Use standard business format. Do not invent facts.
                    
                    Job Description: {job_description}
                    Candidate's Resume: {context}
                    """
                    st.subheader("📝 Your Tailored Cover Letter")
                    def stream_generator():
                        for chunk in chat.stream(template):
                            yield chunk.content
                    st.write_stream(stream_generator())
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def render_ats_analyzer():
    st.markdown('<p class="main-header">🎯 ATS Resume Scorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Check your ATS compatibility against a specific job description.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("1. Upload Resume (PDF)", type=["pdf"], key="ats_uploader")
    with col2:
        job_description = st.text_area("2. Paste Target Job Description", height=200, key="ats_jd")

    if st.button("🚀 Analyze ATS Match"):
        if not uploaded_file or not job_description.strip():
            st.warning("Please provide both a resume and a job description.")
        else:
            with st.spinner("🤖 Analyzing keyword match and parsability..."):
                try:
                    context = extract_text_from_pdf(uploaded_file)
                    template = f"""
                    Act as an expert Applicant Tracking System (ATS). Evaluate the resume against the job description.
                    Job Description: {job_description}
                    Resume: {context}
                    STRICT RULE: Do not hallucinate facts.
                    
                    Provide evaluation EXACTLY in this structure:
                    **Score:** [Number out of 100]
                    **Overall Match:** [%]
                    **Keywords Matched:** [List]
                    **Missing Keywords:** [List]
                    **Readability Score:** [Score out of 100]
                    **ATS Compatibility Score:** [Score out of 100]
                    
                    **Format Analysis:** [2 lines]
                    **Skill Gap Analysis:** [Brief analysis]
                    **Overall Improvement Suggestions:** [2-3 points]
                    **Industry Specific Feedback:** [Brief feedback]
                    """
                    
                    response = chat.invoke(template)
                    result_text = response.content
                    score = extract_score(result_text)

                    st.markdown("---")
                    st.header("📊 Evaluation Results")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        st.plotly_chart(create_pie_chart(score), use_container_width=True)
                        st.metric(label="Overall ATS Score", value=f"{score}/100")
                    with res_col2:
                        st.markdown("### Detailed Analysis")
                        st.markdown(result_text)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def render_career_coach():
    st.markdown('<p class="main-header">💬 AI Career Coach</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive, side-by-side resume reviewing and interview prep.</p>', unsafe_allow_html=True)
    
    # Isolate session state for the coach
    if "coach_history" not in st.session_state:
        st.session_state.coach_history = []
    if "coach_sys_msg" not in st.session_state:
        st.session_state.coach_sys_msg = None
        
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📄 Your Resume")
        uploaded_file = st.file_uploader("Upload resume (PDF) to begin", type=["pdf"], key="coach_uploader")
        
        if uploaded_file:
            display_pdf(uploaded_file)
            
            # Process only if it's a new file
            if "coach_pdf_name" not in st.session_state or st.session_state.coach_pdf_name != uploaded_file.name:
                with st.spinner("Coach is reading your resume..."):
                    context = extract_text_from_pdf(uploaded_file)
                    sys_content = f""" 
                    You are an elite Career Coach and Expert Resume Writer.
                    Provide actionable, industry-standard, and highly personalized career advice. 
                    Tone: Empathetic, encouraging, yet radically candid. 
                    Focus: ATS optimization (XYZ formula), Interview Prep (STAR method), Skill gaps.
                    Candidate Resume: \n{context}
                    """
                    st.session_state.coach_sys_msg = SystemMessage(content=sys_content)
                    st.session_state.coach_history = [] # Reset chat for new resume
                    st.session_state.coach_pdf_name = uploaded_file.name

    with col2:
        st.subheader("Interactive Chat")
        if not uploaded_file:
            st.info("👈 Please upload your resume on the left to activate your AI Coach.")
        else:
            # Display chat history
            chat_container = st.container(height=600)
            with chat_container:
                for msg in st.session_state.coach_history:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    with st.chat_message(role):
                        st.markdown(msg.content)

            # Chat Input
            if prompt := st.chat_input("Ask for resume feedback, interview prep..."):
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                
                st.session_state.coach_history.append(HumanMessage(content=prompt))
                messages = [st.session_state.coach_sys_msg] + st.session_state.coach_history

                with chat_container:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""
                        for chunk in chat.stream(messages):
                            full_response += chunk.content
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                
                st.session_state.coach_history.append(AIMessage(content=full_response))

# --- 5. Sidebar Navigation ---
with st.sidebar:
    st.title("🧞 Resume Genie")
    st.markdown("Select a service below:")
    
    # Use radio buttons for clean navigation
    app_mode = st.radio(
        "Services",
        ["Resume Evaluator", "Cover Letter Generator", "ATS Scorer", "AI Career Coach"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Powered by Llama 3.3 & Streamlit")

# --- 6. Page Routing ---
if app_mode == "Resume Evaluator":
    render_resume_evaluator()
elif app_mode == "Cover Letter Generator":
    render_cover_letter_generator()
elif app_mode == "ATS Scorer":
    render_ats_analyzer()
elif app_mode == "AI Career Coach":
    render_career_coach()