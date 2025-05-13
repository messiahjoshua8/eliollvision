import os
import sys
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Minimal Groq Test")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test_direct():
    """Test Groq API directly using requests instead of the groq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"status": "error", "message": "No GROQ_API_KEY environment variable found"}
    
    try:
        # List models directly with API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            return {
                "status": "success", 
                "message": "Successfully connected to Groq API",
                "models": response.json()["data"][:5]  # Just show first 5 for brevity
            }
        else:
            return {
                "status": "error", 
                "message": f"API error: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.get("/environment")
async def check_environment():
    """Show information about the environment."""
    return {
        "python_version": sys.version,
        "environment_variables": {k: "PRESENT" for k in os.environ if k.startswith("GROQ_")},
        "module_paths": sys.path
    }

if __name__ == "__main__":
    uvicorn.run("minimal_test:app", host="0.0.0.0", port=8081, reload=True) 