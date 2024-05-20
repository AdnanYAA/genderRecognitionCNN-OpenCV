from flask import Flask, render_template, Response, redirect, url_for
from camera import Video

app = Flask(__name__)
video_stream = None
streaming = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    global video_stream, streaming
    if not streaming:
        video_stream = Video()
        streaming = True
    print("Start streaming: ", streaming)
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global video_stream, streaming
    if streaming:
        video_stream.__del__()
        video_stream = None
        streaming = False
    print("Stop streaming: ", streaming)
    return redirect(url_for('index'))

def gen(camera):
    while streaming:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video():
    global streaming, video_stream
    if not streaming or video_stream is None:
        return "Streaming not started", 400
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


