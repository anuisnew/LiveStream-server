# import streamlit as st
# import cv2
# import numpy as np
# import websocket
# import threading
# import time
# from PIL import Image
# from io import BytesIO
# import asyncio
# import nest_asyncio

# # Apply the nest_asyncio patch
# nest_asyncio.apply()

# # Load MobileNet SSD model
# net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# # Define class labels (from COCO dataset)
# classes = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant",
#            "sheep", "sofa", "train", "tvmonitor"]

# # WebSocket settings
# ws_url = "wss://websocket-server-w0tq.onrender.com:443"  # Replace with your WebSocket server URL
# frame = None
# stop_feed = False

# def on_message(ws, message):
#     global frame
#     np_arr = np.frombuffer(message, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# def on_error(ws, error):
#     st.error(f"WebSocket error: {error}")

# def on_close(ws, close_status_code, close_msg):
#     st.warning("WebSocket closed")

# def on_open(ws):
#     st.info("WebSocket connection established")

# def run_websocket():
#     ws = websocket.WebSocketApp(ws_url,
#                                 on_message=on_message,
#                                 on_error=on_error,
#                                 on_close=on_close)
#     ws.on_open = on_open
#     ws.run_forever()

# # Start the WebSocket thread
# ws_thread = threading.Thread(target=run_websocket)
# ws_thread.daemon = True
# ws_thread.start()

# # Streamlit UI
# st.title("ESP32 Cam Stream")

# frame_placeholder = st.empty()

# # Display camera feed
# while not stop_feed:
#     if frame is not None:
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         frame_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
#         buffer = BytesIO()
#         frame_image.save(buffer, format="JPEG")
#         image_bytes = buffer.getvalue()
#         frame_placeholder.image(image_bytes, channels="BGR")
#     time.sleep(0.1)

# st.subheader("Add Pictures for Training")

# uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         image = np.array(Image.open(uploaded_file))
#         st.image(image, caption=uploaded_file.name)

# # Add a stop button to control the feed
# if st.button('Stop Live Feed'):
#     stop_feed = True
#     st.write("Live feed stopped.")


import streamlit as st
import cv2
import numpy as np
import websocket
import threading
import time
from PIL import Image, ImageDraw
from io import BytesIO
import asyncio
import nest_asyncio

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Define class labels (from COCO dataset)
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# WebSocket settings
ws_url = "wss://websocket-server-w0tq.onrender.com:443"  # Replace with your WebSocket server URL
frame = None
stop_feed = False

def on_message(ws, message):
    global frame
    np_arr = np.frombuffer(message, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Ensure frame is decoded as BGR

def on_error(ws, error):
    st.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket closed")

def on_open(ws):
    st.info("WebSocket connection established")

def run_websocket():
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# Start the WebSocket thread
ws_thread = threading.Thread(target=run_websocket)
ws_thread.daemon = True
ws_thread.start()

# Streamlit UI
st.title("ESP32 Cam Stream")

frame_placeholder = st.empty()

# Display camera feed
while not stop_feed:
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for displaying
        frame_image = Image.fromarray(frame_rgb)
        buffer = BytesIO()
        frame_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        frame_placeholder.image(image_bytes, channels="RGB")
    time.sleep(0.1)

st.subheader("Add Pictures for Training")

# Image uploader
uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name)

        # Allow user to draw a box around the object to label
        st.text("Draw a box around the object to label it")
        if st.button("Draw Box"):
            # Code for drawing box can be complex, usually involves JavaScript to handle mouse events.
            # For simplicity, let's assume we allow users to input coordinates manually.
            x1 = st.number_input("x1", min_value=0, max_value=image.width)
            y1 = st.number_input("y1", min_value=0, max_value=image.height)
            x2 = st.number_input("x2", min_value=0, max_value=image.width)
            y2 = st.number_input("y2", min_value=0, max_value=image.height)
            label = st.selectbox("Select Label", classes)

            # Draw the box on the image
            draw = ImageDraw.Draw(image)
            draw.rectangle(((x1, y1), (x2, y2)), outline="red")
            st.image(image, caption=f"Labeled as {label}")

# Add a stop button to control the feed
if st.button('Stop Live Feed'):
    stop_feed = True
    st.write("Live feed stopped.")
