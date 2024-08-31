from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOllama(
    #model = "llama3.1",
    model = "phi3:mini",
    base_url = "http://localhost:11434")

general_agent = Agent(role = "Deep learning Professor",
                      goal = """Provide the solution to the students that are asking machine learning and deep learning questions and give them the answer.""",
                      backstory = """You are an excellent AI/Machine learning/Deep learning professor that likes to give AI/ML/DL questions in a way that everyone can understand""",
                      allow_delegation = False,
                      verbose = True,
                      llm = llm)

if (0):   ##without gradio
    
    task = Task(description="""what is self supervised learning""",
                agent = general_agent,
                expected_output="A brief answer with 50-80 words.")
    crew = Crew(
                agents=[general_agent],
                tasks=[task],
                verbose=True
            )
    result = crew.kickoff()
    print(result)

else:
    import gradio as gr
    def query(payload):
        task = Task(description=payload,
                agent = general_agent,
                expected_output="A brief answer with 50-80 words.")
        crew = Crew(
                    agents=[general_agent],
                    tasks=[task],
                    verbose=True
                )

        result = crew.kickoff()
        
        return result

    def query_fn(input_text):
        resp = query(input_text)
        print(resp.raw)
        return str(resp.raw)

    demo = gr.Interface(fn=query_fn, inputs="text", outputs="label")

    demo.launch()