from flask import Flask, render_template, Response, send_from_directory, jsonify
import time
import json
import os
from audio_processor import start_recording, stop_recording, get_state
from pdf_report import generate_pdf_report

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording_endpoint():
    start_recording()
    return "Recording started"

@app.route('/stop_recording', methods=['POST'])
def stop_recording_endpoint():
    stop_recording()

    transcript, insights = get_state()
    if insights:
        pdf_path = generate_pdf_report(transcript, insights[-1])
        return jsonify({
            "message": "Recording stopped",
            "pdf_url": f"/pdf_reports/{os.path.basename(pdf_path)}"
        })
    return jsonify({"message": "Recording stopped", "pdf_url": None})

@app.route('/pdf_reports/<path:filename>')
def serve_pdf(filename):
    return send_from_directory('pdf_reports', filename)

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint – send NEW partials only (no finals)."""
    def event_stream():
        last_partial = ""          
        last_insight_count = 0    

        while True:
            transcript, insights = get_state()

            payload = {}
            

            if transcript:
                latest = transcript[-1]
                if latest.startswith("PARTIAL:") and latest != last_partial:
                    payload["transcript"] = [latest]
                    last_partial = latest



            if len(insights) > last_insight_count:
                payload["insights"] = insights[last_insight_count:]
                last_insight_count = len(insights)



            if payload:
                yield f"data: {json.dumps(payload)}\n\n"

            time.sleep(0.5)

    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    os.makedirs('pdf_reports', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
