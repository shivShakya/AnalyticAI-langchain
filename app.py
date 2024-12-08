from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import re
import os
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fig_to_base64(fig):
    import io
    import base64

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')

def get_fig_from_code(code):
    local_variables = {}
    exec(code, local_variables)
    return local_variables["plt"]

@app.post("/upload-file/")
async def upload_file(file: UploadFile):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        return {"columns": df.columns.tolist(), "data": df.head().to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GraphRequest(BaseModel):
    user_input: str
    data: list
    name_of_file: str

@app.post("/create-graph/")
async def create_graph(request: GraphRequest):
    """Creates a graph based on user input and uploaded data."""
    df = pd.DataFrame(request.data)
    df_5_rows = df.head()
    csv_string = df_5_rows.to_string(index=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a data visualization expert and you use your favorite graphing library matplotlib only. "
                "Suppose the data is provided as a {name_of_file} file. Here are the first 5 rows of the dataset: {data}. "
                "Follow the user's indications when creating the graph."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | model
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=request.user_input)],
            "data": csv_string,
            "name_of_file": request.name_of_file,
        }
    )
    result_output = response.content
    code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", result_output, re.DOTALL)
    
    if code_block_match:
         code_block = code_block_match.group(1).strip()
         cleaned_code = re.sub(r"(?m)^\s*plt\.show\(\)\s*$", "", code_block)
         print(cleaned_code)
    
         fig = get_fig_from_code(cleaned_code)
         base64_image = fig_to_base64(fig)
         
         return JSONResponse(content={"figure": base64_image, "code": result_output})
    else:
         return JSONResponse(content={"error": "No code block found", "response": result_output})

# Add this block to run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
