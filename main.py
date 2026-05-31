import uvicorn
from api.config import settings

def main():
    """
    Main entry point for the application
    """
    # Run the application
    uvicorn.run(
        "api.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 