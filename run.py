# run.py  –  entry-point for the .exe
import threading, time, os, sys
from streamlit.web import bootstrap          # 🆕 programmatic launch
import webview                               # pip install pywebview==4.*
import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"    # suppress browser tab
os.environ["STREAMLIT_THEME_BASE"]     = "light"    # force light theme
os.environ["STREAMLIT_SERVER_PORT"]    = "8501"     # match the port in webview

APP_FILE = os.path.join(os.path.dirname(__file__), "app.py")
PORT      = 8501

def start_streamlit():
    bootstrap.run(APP_FILE, "", [], {
    "server.headless": True,
    "server.port": 8501,
    "theme.base": "light"
})
      # launch without CLI

threading.Thread(target=start_streamlit, daemon=True).start()
time.sleep(2)                                # give server a head-start

webview.create_window("VegaDelta Options Analytics",
                      f"http://localhost:{PORT}",
                      width=1200, height=800,
                      zoomable=True, resizable=True)
webview.start()
