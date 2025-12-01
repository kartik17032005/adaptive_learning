from fastapi import FastAPI
from main import AdaptiveLearningAssistant, SubjectType

app = FastAPI()
assistant = AdaptiveLearningAssistant()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/learn/{student_id}/{subject}")
async def learn(student_id: str, subject: str, query: str):
    subj = SubjectType(subject.lower())
    return await assistant.process_learning_request(student_id, subj, query)
