import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import os

class ReloadHandler(FileSystemEventHandler):
    def __init__(self, script):
        self.script = script
        self.process = None
        self.start_script()

    def start_script(self):
        if self.process:
            self.process.kill()
        print(f"🚀 Starting {self.script}")
        self.process = subprocess.Popen([sys.executable, self.script])

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"🔄 Detected change in {event.src_path}, restarting...")
            self.start_script()

if __name__ == "__main__":
    script = "app/app.py"  # Your main entry file
    handler = ReloadHandler(script)
    observer = Observer()
    observer.schedule(handler, path=".", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
