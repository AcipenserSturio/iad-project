"""
Web service for model inference.
"""
import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from keras.saving import load_model
from pydantic.dataclasses import dataclass

from src.symbiotic_model import SymbioticNN
from src.text_preprocessing import preprocess

ROOT_PATH = Path(__file__).parent
ASSETS_PATH = ROOT_PATH / 'assets'


def init_application() -> tuple[FastAPI, SymbioticNN]:
    """
    Initialize core application.

    Run: uvicorn reference_service.server:app --reload
    """
    server = FastAPI()

    data_path = ROOT_PATH / 'data'

    server.mount("/assets", StaticFiles(directory=str(ASSETS_PATH)), name='assets')

    model1 = load_model(data_path / 'full_con_nn.h5')
    model2 = load_model(data_path / 'cnn_p.h5')

    model = SymbioticNN(model1, model2, 0.6)

    return server, model


app, symb_model = init_application()


@dataclass
class Query:
    """
    Abstraction class with text field
    """
    text: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint.
    """
    templates = Jinja2Templates(directory=str(ASSETS_PATH))
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Infer the query in web-site
    """
    return {"infer": symb_model.infer(preprocess(query.text))}


if __name__ == "__main__":
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=True)
