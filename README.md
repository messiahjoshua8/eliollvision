# Medical Supply Vision API

A real-time computer vision API that uses Groq and Meta's Llama 4 Scout model to detect and analyze medical supplies from camera images.

## Features

- **Real-time Analysis**: Continuously processes camera frames to identify medical supplies
- **Structured Data Extraction**: Extracts specific fields like Implant Type, GTIN, LOT numbers, expiration dates, etc.
- **Web Interface**: Includes a user-friendly web interface to interact with the camera and view results
- **Image Storage**: Option to save captured images for record-keeping and inventory management
- **API Endpoints**: RESTful API for integration with other applications
- **Dockerfile**: Ready to deploy on platforms like Railway

## Getting Started

### Prerequisites

- Python 3.11+
- A Groq API key

### Local Development

1. Clone the repository
2. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python app.py
   ```
5. Open `http://localhost:8000` in your browser

### Deployment on Railway

1. Create a new project on Railway
2. Connect your repository
3. Add the GROQ_API_KEY environment variable
4. Deploy!

## API Endpoints

- **GET /** - Web interface
- **GET /health** - Health check endpoint
- **POST /analyze/frame** - Analyze an image frame
  - Parameters:
    - `image`: File upload (required)
    - `save_image`: Boolean (default: false)

## How It Works

1. The web interface captures frames from the camera
2. Frames are sent to the backend API
3. The API sends the image to Groq's Llama 4 Scout model
4. The model analyzes the image and returns detailed text
5. Text is parsed to extract structured data
6. Results are displayed in real-time on the web interface

## Use Cases

- Medical supply inventory management
- Surgical equipment verification
- Expiration date tracking
- Medical device authentication

## Limitations

- Requires a good quality camera for accurate text recognition
- Certain lighting conditions may affect performance
- Language support is currently limited to English 