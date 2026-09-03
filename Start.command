#!/bin/bash
# Double-click this to start Vehicle Counter.
#
# macOS may refuse the first time ("cannot be opened because it is from an
# unidentified developer"). Right-click the file, choose Open, then Open again.
# You only have to do that once.

cd "$(dirname "$0")"

echo ""
echo "  Vehicle Counter"
echo "  ==============="
echo ""

if ! python3 -c "import flask, cv2, ultralytics" 2>/dev/null; then
  echo "  First run — installing dependencies. This takes a few minutes."
  echo ""
  python3 -m pip install -r requirements.txt || {
    echo ""
    echo "  Install failed. Try running this in Terminal:"
    echo "    python3 -m pip install -r requirements.txt"
    read -n 1 -s -r -p "  Press any key to close."
    exit 1
  }
  echo ""
fi

# Open the browser once the server is actually listening, not before
( for _ in $(seq 1 40); do
    if curl -s -o /dev/null http://127.0.0.1:8000/ ; then
      open http://127.0.0.1:8000/ ; break
    fi
    sleep 0.5
  done ) &

echo "  Starting on http://127.0.0.1:8000"
echo "  Close this window to stop."
echo ""
python3 -u app.py
