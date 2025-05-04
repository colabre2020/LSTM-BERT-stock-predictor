from fastapi import FastAPI, HTTPException, Depends
from app.routes import prediction, auth
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stock Forecast API")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(prediction.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])

@app.get("/")
def root():
    return {"message": "Welcome to Stock Forecast API"}
