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
from groq import Groq
from dotenv import load_dotenv

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
try:
    # Check for various possible environment variable names
    default_api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key") or os.getenv("Groq_Api_Key")
    if default_api_key:
        print(f"Initializing Groq client with API key (length: {len(default_api_key)})")
        default_client = Groq(api_key=default_api_key)
    else:
        print("No Groq API key found in environment variables")
except Exception as e:
    print(f"Error initializing default Groq client: {e}")

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
    """Send image to Groq LLM for analysis."""
    # Use custom API key if provided, otherwise use default client
    client = default_client
    if api_key:
        try:
            print(f"Using custom API key provided in request header (length: {len(api_key)})")
            client = Groq(api_key=api_key)
        except Exception as e:
            print(f"Error initializing custom Groq client: {str(e)}")
    
    if client is None:
        print("No Groq client available. Default client initialized: ", default_client is not None)
        raise HTTPException(status_code=500, detail="Groq client not initialized. Please check your API key.")
    
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
    
    try:
        # Call the Groq API
        print("Calling Groq API with model: meta-llama/llama-4-scout-17b-16e-instruct")
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        
        # Return the model's response
        return completion.choices[0].message.content
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
    
    # Check if a custom API key was provided
    if x_groq_api_key:
        try:
            print(f"Health check with custom API key (length: {len(x_groq_api_key)})")
            test_client = Groq(api_key=x_groq_api_key)
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
        if not default_client:
            error_details = "Default Groq client not initialized"
    
    # Check environment variables (without revealing the actual keys)
    env_vars = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY") is not None,
        "groq_api_key": os.getenv("groq_api_key") is not None,
        "Groq_Api_Key": os.getenv("Groq_Api_Key") is not None
    }
    
    return {
        "status": "ok", 
        "groq_client": client_status,
        "api_key_source": api_key_info,
        "environment_vars": env_vars,
        "error": error_details,
        "opencv_available": CV2_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True) 