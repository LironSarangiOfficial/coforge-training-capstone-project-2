import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, YoutubeVideoSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_pipeline import medical_rag_tool


search_tool = SerperDevTool()

youtube_tool = YoutubeVideoSearchTool()

llm = LLM(
	model='gemini/gemini-2.5-flash',
	temperature=0.4,   # Low temperature for medical data precision
	max_tokens=2048
)

retriever_agent = Agent(
    role='Medical Data Retriever',
    goal='Extract the most relevant medical articles, patient notes, and guidelines for {query}',
    backstory="""You are an expert medical librarian. Your strength lies in your ability 
    to navigate complex medical databases and retrieve precise documentation. You provide 
    the raw evidence that the rest of the team relies on.""",
    tools=[medical_rag_tool, search_tool, youtube_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

diagnosis_agent = Agent(
    role='Diagnosis Support Specialist',
    goal='Analyze the retrieved evidence to suggest possible medical conditions.',
    backstory="""You are a diagnostic specialist. You take raw medical data and 
    identify clinical patterns, highlighting potential conditions or risks based 
    strictly on the evidence provided by the Retriever Agent.""",
    tools=[medical_rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=True
)

consultant_agent = Agent(
    role='Medical Consultation Expert',
    goal='Synthesize the research and diagnostic suggestions into a grounded, professional response.',
    backstory="""You are a senior medical consultant known for your empathy and clarity. 
    Your job is to take the evidence and suggestions from your colleagues and draft a 
    response that is easy to understand, medically grounded, and actionable.""",
    tools=[medical_rag_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# Defining the Tasks

retrieval_task = Task(
    description="""Search the medical database for documents related to: {query}. 
    Focus on extracting specific symptoms, medical history, and clinical guidelines.""",
    expected_output="A structured list of raw medical evidence and relevant transcript segments.",
    agent=retriever_agent
)

# 2. Diagnosis Support Task
diagnosis_task = Task(
    description="""Using the evidence gathered, identify clinical patterns and 
    suggest 3-4 possible conditions. Highlight specific data points that support each suggestion.""",
    expected_output="A diagnostic report listing potential conditions with evidence-based justifications.",
    agent=diagnosis_agent,
    context=[retrieval_task] # Links back to Retriever's output
)

# 3. Consultation Task
consultation_task = Task(
    description="""Synthesize the findings and diagnostic suggestions into a final response. 
    Address the user's query directly and include relevant YouTube resources for clarity.""",
    expected_output="A grounded, empathetic medical summary with actionable advice and visual references.",
    agent=consultant_agent,
    context=[retrieval_task, diagnosis_task] # Links back to both previous outputs
)


medical_crew = Crew(
    agents=[retriever_agent, diagnosis_agent, consultant_agent],
    tasks=[retrieval_task, diagnosis_task, consultation_task],
    process=Process.sequential # Ensures tasks execute in order
    
)


