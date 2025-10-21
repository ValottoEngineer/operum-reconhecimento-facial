# bridge_server.py
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# estado simples de demo
STATE = {"status": "idle"}  # idle | pending | approved | rejected | cancelled

@app.post("/start")
def start():
    STATE["status"] = "pending"
    return jsonify(STATE), 200

@app.get("/status")
def status():
    # opcional: se approved/rejected for lido, você pode manter até o front limpar
    return jsonify(STATE), 200

@app.post("/approve")
def approve():
    STATE["status"] = "approved"
    return jsonify(STATE), 200

@app.post("/reject")
def reject():
    STATE["status"] = "rejected"
    return jsonify(STATE), 200

@app.post("/cancel")
def cancel():
    STATE["status"] = "cancelled"
    return jsonify(STATE), 200

if __name__ == "__main__":
    # use a mesma porta configurada no httpFace.ts (5001)
    app.run(host="127.0.0.1", port=5001)
