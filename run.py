"""Main entry point for the whisky recognition system."""

import argparse
import sys

def main():
    """Process command line arguments and run the specified module."""
    parser = argparse.ArgumentParser(description="Whisky Recognition System")
    parser.add_argument('module', choices=['api', 'demo'], 
                       help='Module to run: "api" for REST API server, "demo" for demonstration')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Port number for API server (only with "api" module)')
    parser.add_argument('--image', type=str, default='test_image.jpg', 
                       help='Path to test image (only with "demo" module)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.module == 'api':
        from api import WhiskyRecognitionAPI
        print(f"Starting API server on port {args.port}...")
        api = WhiskyRecognitionAPI()
        api.run(port=args.port)
        
    elif args.module == 'demo':
        from demo import run_demo
        run_demo(args.image)
        
if __name__ == "__main__":
    main()