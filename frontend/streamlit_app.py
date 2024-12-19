import streamlit as st
import pandas as pd
import requests

SERVER_WS = "ws://127.0.0.1:8000/ws"
SERVER_API = "http://127.0.0.1:8000"
user_id = "user_123"
ids = []

st.title("Audio Recording and Song Recommendation System")

audio_recorder = f"""
<script>
    let mediaRecorder;
    let socket;
    let isRecording = false;

    function toggleRecording() {{
        if (!isRecording) {{
            startRecording();
        }} else {{
            stopRecording();
        }}
    }}

    async function startRecording() {{
        isRecording = true;
        const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        mediaRecorder = new MediaRecorder(stream);
        socket = new WebSocket("{SERVER_WS}?user_id={user_id}");

        socket.onopen = () => {{
            document.getElementById("status").innerText = "🔴 Recording started...";
        }};

        mediaRecorder.ondataavailable = (event) => {{
            if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {{
                socket.send(event.data);
            }}
        }};

        mediaRecorder.start(500);
    }}

    function stopRecording() {{
        isRecording = false;
        mediaRecorder.stop();
        socket.send("END");
        socket.close();
        document.getElementById("status").innerText = "⏹ Recording stopped.";
    }}

    window.addEventListener("load", () => {{
        const recordButton = document.getElementById("recordButton");
        recordButton.addEventListener("click", toggleRecording);
    }});
</script>

<button id="recordButton" style="font-size: 20px; padding: 10px; margin-top: 20px;">
    Start / Stop Recording
</button>
<p id="status" style="font-size: 18px; margin-top: 10px;">⏹ Recording not started.</p>
"""

st.components.v1.html(audio_recorder, height=200)

def fetch_json(endpoint, params=None):
    response = requests.get(f"{SERVER_API}/{endpoint}", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code}")
        return None

st.subheader("Get Similar Songs")
similar_data = None
if st.button("Fetch Similar Songs"):
    similar_data = fetch_json("similar", {"user_id": user_id})
    if similar_data:
        st.success("Similar Songs Fetched Successfully")
        # st.json(similar_data)
        df = pd.DataFrame(similar_data)
        st.subheader("Similar Songs (DataFrame):")
        st.dataframe(df)
        ids = df["id"].tolist()

        default_songs = ",".join(ids)
        json_data = fetch_json("recommend", {"songs": default_songs})
        if json_data:
            st.success("Recommendations Fetched Successfully")
            # st.json(json_data)
            df = pd.DataFrame(json_data)
            st.subheader("Recommendations (DataFrame):")
            st.dataframe(df)
