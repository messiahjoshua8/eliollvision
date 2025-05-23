import os
import io
import base64
import time
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
# Import groq conditionally, with a more compatible approach
try:
    from groq import Groq
    GROQ_IMPORT_ERROR = None
    
    # Apply preemptive monkey patch to Groq.__init__ to handle proxies parameter
    # This ensures it's patched before any client initialization
    original_init = Groq.__init__
    
    def patched_init(self, *args, **kwargs):
        # Always remove proxies parameter if it exists
        if 'proxies' in kwargs:
            print("Preemptively removing 'proxies' parameter")
            del kwargs['proxies']
        return original_init(self, *args, **kwargs)
    
    # Apply the monkey patch
    Groq.__init__ = patched_init
    print("Applied preemptive monkey patch to Groq.__init__")
    
except Exception as e:
    GROQ_IMPORT_ERROR = str(e)
    Groq = None
from dotenv import load_dotenv
import requests  # Add this import at the top

# Define a fallback class that mimics the Groq client but uses requests directly
class FallbackGroqClient:
    """A fallback implementation that mimics the Groq client interface but uses requests directly."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Create a models property that has a list method
        class Models:
            def __init__(self, client):
                self.client = client
            
            def list(self):
                response = requests.get(f"{self.client.base_url}/models", headers=self.client.headers)
                if response.status_code != 200:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
                return self.format_response(response.json())
            
            def format_response(self, json_data):
                # Convert the JSON response to a structure that matches the expected Groq client format
                class ModelListResponse:
                    def __init__(self, data):
                        self.data = data
                return ModelListResponse(json_data["data"])
                
        self.models = Models(self)
        
        # Create a chat property that has a completions subproperty with a create method
        class Completions:
            def __init__(self, client):
                self.client = client
            
            def create(self, model, messages, temperature=0.7, max_tokens=1024, top_p=1, stream=False):
                data = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stream": stream
                }
                response = requests.post(
                    f"{self.client.base_url}/chat/completions", 
                    headers=self.client.headers,
                    json=data
                )
                
                if response.status_code != 200:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
                    
                # Format the response to match what the Groq client would return
                return self.format_response(response.json())
            
            def format_response(self, json_data):
                # Convert the JSON response to a structure that matches the expected Groq client format
                class Choice:
                    def __init__(self, message):
                        self.message = message
                
                class Message:
                    def __init__(self, content, role):
                        self.content = content
                        self.role = role
                
                class CompletionResponse:
                    def __init__(self, choices):
                        self.choices = choices
                
                # Create the Message object from the response
                message = Message(
                    json_data["choices"][0]["message"]["content"],
                    json_data["choices"][0]["message"]["role"]
                )
                
                # Create the Choice object with the message
                choice = Choice(message)
                
                # Return a CompletionResponse with the choices
                return CompletionResponse([choice])
        
        # Create the chat.completions nested attribute
        self.chat = type('Chat', (), {'completions': Completions(self)})()

# Define the create_groq_client function before it's used
def create_groq_client(api_key):
    """Create a Groq client while handling version differences."""
    if Groq is None:
        raise ValueError(f"Groq module import failed: {GROQ_IMPORT_ERROR}")
        
    try:
        # Create the client
        client = Groq(api_key=api_key)
        # Test that it works by accessing a property
        _ = client.base_url
        return client
    except Exception as e:
        print(f"Error creating Groq client with standard approach: {str(e)}")
        # Try using our fallback implementation
        try:
            print("Using fallback Groq client implementation")
            fallback_client = FallbackGroqClient(api_key)
            # Test that it works
            _ = fallback_client.base_url
            return fallback_client
        except Exception as e2:
            print(f"Error creating fallback Groq client: {str(e2)}")
            raise

# Try to import OpenCV, but continue if it fails
try:
    import cv2
    CV2_AVAILABLE = True
except (ImportError, AttributeError):
    print("WARNING: OpenCV (cv2) import failed. Some functionality may be limited.")
    CV2_AVAILABLE = False

# Load environment variables
load_dotenv()

app = FastAPI(title="Medical Supplies Vision API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Groq client with environment variable
default_client = None
if Groq is not None:
    try:
        # Check for the environment variable
        default_api_key = os.getenv("GROQ_API_KEY")
        if default_api_key:
            print(f"Found API key in environment variables (length: {len(default_api_key)})")
            # Create client with our wrapper function
            default_client = create_groq_client(default_api_key)
            # Test that the client works
            print("Testing Groq client initialization...")
            _ = default_client.base_url  # Just access a property to verify initialization
            print("Groq client initialized successfully")
        else:
            print("No GROQ_API_KEY found in environment variables")
    except Exception as e:
        print(f"Error initializing default Groq client: {str(e)}")
else:
    print(f"Groq module import failed: {GROQ_IMPORT_ERROR}")

class ImageAnalysisResponse(BaseModel):
    detected_text: str
    implant_type: Optional[str] = None
    gtin: Optional[str] = None
    lot: Optional[str] = None
    expiration_date: Optional[str] = None
    sterile: Optional[str] = None
    size: Optional[str] = None
    barcode: Optional[str] = None
    quantity: Optional[str] = None
    timestamp: str
    image_id: str

def extract_medical_supply_info(text: str) -> dict:
    """Extract structured information from the detected text."""
    result = {
        "implant_type": None,
        "gtin": None,
        "lot": None,
        "expiration_date": None,
        "sterile": None,
        "size": None,
        "barcode": None,
        "quantity": None
    }
    
    # Look for specific patterns in the text
    lines = text.split('\n')
    for line in lines:
        line = line.strip().lower()
        if "implant type:" in line:
            result["implant_type"] = line.split(":", 1)[1].strip()
        elif "gtin:" in line:
            result["gtin"] = line.split(":", 1)[1].strip()
        elif "lot:" in line:
            result["lot"] = line.split(":", 1)[1].strip()
        elif "exp:" in line or "expiration date:" in line or "expiration:" in line:
            result["expiration_date"] = line.split(":", 1)[1].strip()
        elif "sterile" in line:
            result["sterile"] = line.split(":", 1)[1].strip() if ":" in line else "Yes" if "yes" in line else "No"
        elif "size:" in line:
            result["size"] = line.split(":", 1)[1].strip()
        elif "barcode:" in line or "upc:" in line or "ean:" in line:
            result["barcode"] = line.split(":", 1)[1].strip()
        elif "qty:" in line or "quantity:" in line or "count:" in line:
            result["quantity"] = line.split(":", 1)[1].strip()
            
    # Try to extract barcode if not found but looks like it's in the text
    if result["barcode"] is None:
        # Look for numeric strings that could be barcodes (12-14 digits)
        for line in lines:
            words = line.split()
            for word in words:
                # Clean up the word to just digits
                digits_only = ''.join(c for c in word if c.isdigit())
                if len(digits_only) >= 12 and len(digits_only) <= 14:
                    result["barcode"] = digits_only
                    break
            if result["barcode"]:
                break
                
    # Try to extract quantity if not found
    if result["quantity"] is None:
        for line in lines:
            if "pack" in line.lower() or "count" in line.lower() or "pcs" in line.lower() or "pieces" in line.lower():
                # Look for digits near these words
                digits = ''.join(c for c in line if c.isdigit())
                if digits:
                    result["quantity"] = digits
                    break
            
    return result

def analyze_image_with_llm(image_data: bytes, api_key: Optional[str] = None) -> str:
    """Send image to Groq LLM for analysis using direct API calls."""
    # Use provided API key or fallback to environment variable
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="No Groq API key provided or found in environment")
    
    # Convert image to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare the message for the model
    messages = [
        {
            "role": "system",
            "content": "You are a specialized medical inventory AI assistant. Your task is to identify medical supplies in images and extract key information like Implant Type, GTIN, LOT number, Expiration Date, Sterility status, Size, Barcode, and Quantity. Provide the information in a clear, structured format with one piece of information per line. If you can't identify something, indicate it's not visible."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What medical supply is shown in this image? Extract all visible information including: Implant Type, GTIN, LOT number, Expiration Date, Sterility status, Size, Barcode, and Quantity. Format the output clearly with one piece of information per line. If there's a barcode visible, please include the number. If there's any indication of quantity or package count, include that as well."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # Prepare the request payload
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 1
    }
    
    # Prepare headers with authorization
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Call the Groq API directly
        print("Calling Groq API with model: meta-llama/llama-4-scout-17b-16e-instruct")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        # Check for errors
        if response.status_code != 200:
            error_detail = f"API error: {response.status_code} - {response.text}"
            print(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)
        
        # Parse and return the model's response
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
        
    except Exception as e:
        error_message = str(e)
        print(f"Groq API error: {error_message}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {error_message}")

@app.get("/", response_class=HTMLResponse)
async def get_web_app():
    """Serve the main web application."""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/test", response_class=HTMLResponse)
async def get_test_page():
    """Serve the test page for simple API testing."""
    with open("static/test.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/webcam", response_class=HTMLResponse)
async def get_webcam_page():
    """Serve the webcam interface for real-time analysis."""
    with open("static/webcam.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/analyze/frame", response_model=ImageAnalysisResponse)
async def analyze_frame(
    image: UploadFile = File(...),
    save_image: bool = Form(False),
    x_groq_api_key: Optional[str] = Header(None)
):
    """Analyze a single frame/image from the camera."""
    start_time = time.time()
    
    # Read and process the image
    image_data = await image.read()
    
    # Generate a unique ID for this image analysis
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_id = f"img_{timestamp}"
    
    # Save the image if requested
    if save_image:
        os.makedirs("saved_images", exist_ok=True)
        image_path = f"saved_images/{image_id}.jpg"
        with open(image_path, "wb") as f:
            f.write(image_data)
    
    # Analyze the image using Groq LLM
    detected_text = analyze_image_with_llm(image_data, x_groq_api_key)
    
    # Extract structured information
    info = extract_medical_supply_info(detected_text)
    
    # Prepare the response
    response = ImageAnalysisResponse(
        detected_text=detected_text,
        implant_type=info["implant_type"],
        gtin=info["gtin"],
        lot=info["lot"],
        expiration_date=info["expiration_date"],
        sterile=info["sterile"],
        size=info["size"],
        barcode=info["barcode"],
        quantity=info["quantity"],
        timestamp=datetime.now().isoformat(),
        image_id=image_id
    )
    
    print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    return response

@app.get("/health")
async def health_check(x_groq_api_key: Optional[str] = Header(None)):
    """Health check endpoint."""
    client_status = False
    api_key_info = None
    error_details = None
    import_status = "available" if Groq is not None else "failed"
    
    # Check if a custom API key was provided
    if x_groq_api_key and Groq is not None:
        try:
            print(f"Health check with custom API key (length: {len(x_groq_api_key)})")
            test_client = create_groq_client(x_groq_api_key)
            
            # Just try to access a property to verify it's initialized properly
            _ = test_client.base_url
            client_status = True
            api_key_info = "custom"
        except Exception as e:
            error_details = str(e)
            print(f"Custom API key health check failed: {error_details}")
    else:
        # Check the default client
        client_status = default_client is not None
        api_key_info = "default" if default_client else "none"
        if Groq is None:
            error_details = f"Groq module import error: {GROQ_IMPORT_ERROR}"
        elif not default_client:
            error_details = "Default Groq client not initialized"
    
    # Check environment variables (without revealing the actual keys)
    env_vars = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY") is not None,
    }
    
    return {
        "status": "ok", 
        "groq_client": client_status,
        "api_key_source": api_key_info,
        "environment_vars": env_vars,
        "error": error_details,
        "groq_import": import_status,
        "opencv_available": CV2_AVAILABLE
    }

@app.get("/api-key-test")
async def test_api_key(key: Optional[str] = None):
    """Test a specific API key or the environment variable."""
    if key is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return {"status": "error", "message": "No API key provided and no GROQ_API_KEY in environment"}
    
    if Groq is None:
        return {"status": "error", "message": f"Groq module import error: {GROQ_IMPORT_ERROR}"}
    
    # Try to create a client and make a simple API call
    try:
        print(f"Testing API key (length: {len(key)})")
        client = create_groq_client(key)
        
        # Try to list models (a simple API call)
        print("Making test API call...")
        try:
            models = client.models.list()
            return {"status": "success", "message": "API key is valid", "models_available": len(models.data)}
        except Exception as e:
            return {"status": "error", "message": f"API key accepted but API call failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to initialize client: {str(e)}"}

# Add this endpoint after the api-key-test endpoint
@app.get("/direct-test")
async def direct_test(key: Optional[str] = None):
    """Test Groq API directly using requests instead of the groq client."""
    if key is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return {"status": "error", "message": "No API key provided and no GROQ_API_KEY in environment"}
    
    try:
        # List models directly with API call
        headers = {
            "Authorization": f"Bearer {key}",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True) 