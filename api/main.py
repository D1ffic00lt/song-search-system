import warnings
import subprocess
import uuid
import pandas as pd
import uvicorn
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from io import BytesIO

from utils import *

warnings.filterwarnings("ignore")

app = FastAPI()
manager = ConnectionManager()
text_data = {"user_id": "i'm just a freak"}
rec_model = load_rec_model("model/nn_model.model")
songs_data = pd.read_csv("./recommendation_system/dataset/vectors_data.csv")
songs_data = reduce_memory_usage(songs_data)
word2vec = Word2VecModel("./model/word2vec_lyrics.model")
speech2text = Speech2TextModel("openai/whisper-tiny")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = None):
    await manager.connect(websocket)
    audio_data = BytesIO()

    if not user_id:
        await websocket.close(code=1008, reason="User ID is required")
        return

    try:
        while True:
            try:
                message = await websocket.receive()
                if "bytes" in message:
                    data = message["bytes"]
                    if data == b"END":
                        break
                    audio_data.write(data)
                    print(f"Received {len(data)} bytes")
                elif "text" in message:
                    print(f"Received text: {message['text']}")
            except RuntimeError as e:
                print("WebSocket connection closed:", e)
                break
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        manager.disconnect(websocket)
        if audio_data.getbuffer().nbytes > 0:
            random_id = uuid.uuid4().hex
            raw_audio_path = f"raw_audio_{random_id}.webm"
            with open(raw_audio_path, "wb") as f:
                f.write(audio_data.getvalue())

            output_audio_path = f"received_audio_{random_id}.mp3"
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i", raw_audio_path,
                        "-vn",
                        "-ar", "44100",
                        "-ac", "2",
                        "-b:a", "192k",
                        output_audio_path
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                text_data[user_id] = speech2text.predict(output_audio_path)
                print(f"Transcription for {user_id}: {text_data[user_id]}")
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg Error: {e}")
            finally:
                os.remove(raw_audio_path)
                if os.path.exists(output_audio_path):
                    os.remove(output_audio_path)


@app.get("/similar")
async def similar(user_id: str):
    if user_id not in text_data:
        return {"error": "No data for this user"}
    text = text_data[user_id]
    vector = word2vec(text)
    if vector is None:
        return {"error": "No vector for this text"}
    return get_similar_songs(vector, songs_data)


@app.get("/recommend")
async def recommend(songs: str):
    if not songs:
        return {"error": "No songs provided"}
    return recommend_songs(songs.split(","), songs_data, rec_model)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
