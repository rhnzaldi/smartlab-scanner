import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient
from main import app

# Create a dummy image
img = np.zeros((640, 480, 3), dtype=np.uint8)
# add a "face" so it doesn't fail fast
cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

_, buffer = cv2.imencode('.jpg', img)
base64_str = base64.b64encode(buffer).decode('utf-8')

client = TestClient(app)

print("Sending request to /api/face/enroll...")
response = client.post("/api/face/enroll", json={
    "nim": "J0303211000",
    "nama": "Test User",
    "image_base64": base64_str
})

print("Status Code:", response.status_code)
print("Response:", response.json() if response.headers.get("content-type") == "application/json" else response.text)
