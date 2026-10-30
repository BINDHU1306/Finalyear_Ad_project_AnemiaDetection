# Run HemoGlow AI - Complete Anemia Screening Application
import os
import sys
import webbrowser
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.backend.app import app

def open_browser(port):
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"  HEMOGLOW AI SCREENING SYSTEM IS LIVE!")
    print(f"  Web Interface URL: {url}")
    print(f"  Malla Reddy University - Major Project Phase-1")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Notice: Could not automatically open browser ({e}). Please navigate to {url}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host='127.0.0.1', port=port, debug=False)
