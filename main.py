"""
main.py – Launch script for Smart Industrial Energy AI Full-Stack Platform
Runs the FastAPI web server on http://localhost:8000
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("\n" + "="*70)
    print("⚡ Smart Industrial Energy AI – Web Platform")
    print("="*70)
    print("🌐 Full-Stack Modern Web UI: http://localhost:8000")
    print("📄 API Documentation (Swagger): http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
