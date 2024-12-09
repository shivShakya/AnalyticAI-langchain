from fastapi import FastAPI, UploadFile, HTTPException
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

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the AI model
model = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

# FastAPI app initialization
app = FastAPI()

# Initialize global dataframe variable
dataframe = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 string."""
    import io
    import base64

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')

def execute_code_with_dataframe(code, dataframe):
    """Executes the given code with the dataframe in scope."""
    try:
        import matplotlib.pyplot as plt
        safe_dataframe = dataframe.copy(deep=True)
        local_variables = {"dataframe":  safe_dataframe, "plt": plt}
        exec(code, {}, local_variables)
        return local_variables.get("plt")
    except Exception as e:
        raise RuntimeError(f"Error executing code: {e}")

@app.post("/upload-file/")
async def upload_file(file: UploadFile):
    """Handles file upload and sets the global dataframe."""
    global dataframe
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            dataframe = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xls", ".xlsx")):
            dataframe = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        return {"columns": dataframe.columns.tolist(), "data": dataframe.head().to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

class GraphRequest(BaseModel):
    user_input: str
    name_of_file: str

@app.post("/create-graph/")
async def create_graph(request: GraphRequest):
    """Creates a graph based on user input and uploaded data."""
    global dataframe
    if dataframe is None:
        raise HTTPException(status_code=400, detail="No dataframe found. Please upload a file first.")

    try:
        df_preview = dataframe.head()
        csv_string = df_preview.to_string(index=True)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a data visualization expert and use matplotlib as your graphing library. "
                    "The dataset is provided as a variable named 'dataframe' and 'plt' as matplotlib reference use it. Here are the first 5 rows of the dataframe: {data}. "
                    "Follow the user's instructions to create the graph."
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
        print(result_output)
        # Extract Python code from the response
        code_block_match = re.search(r"```(?:[Pp]ython)?(.*?)```", result_output, re.DOTALL)
        if code_block_match:
            code_block = code_block_match.group(1).strip()
            cleaned_code = re.sub(r"(?m)^\s*plt\.show\(\)\s*$", "", code_block)

            # Execute the code with the global dataframe
            fig = execute_code_with_dataframe(cleaned_code, dataframe)
            base64_image = fig_to_base64(fig)

            return JSONResponse(content={"figure": base64_image, "code": result_output})
        else:
            return JSONResponse(content={"error": "No valid code block found in the AI response", "response": result_output})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
