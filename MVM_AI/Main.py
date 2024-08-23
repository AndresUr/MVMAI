from fastapi import FastAPI
from Backend import api

#
app = FastAPI()

# Incluye el router
app.include_router(api.router)