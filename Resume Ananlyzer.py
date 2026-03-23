from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
import docx
import ollama
from werkzeug.utils import secure_filename
from io import BytesIO

ALLOWED_EXTENSIONS = {"pdf", "docx"}

app = Flask(__name__)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_bytes(file_bytes, filename):
    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(file_bytes))
            return "".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            return f"[Error reading PDF: {e}]"
    elif filename.endswith(".docx"):
        try:
            doc = docx.Document(BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            return f"[Error reading DOCX: {e}]"
    return ""

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        requirements = request.form.get("requirements", "").strip()
        files = request.files.getlist("files")

        if not requirements or not files:
            return jsonify({"error": "Requirements and files are required"}), 400

        results = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_bytes = file.read()
                text = extract_text_from_bytes(file_bytes, filename)

                if not text.strip():
                    results.append({
                        "file": filename,
                        "result": "(No extractable text found, possibly scanned)"
                    })
                    continue

                response = ollama.chat(
                    model="llama3.2:1b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an HR recruiter at a reputed company. "
                                "You must strictly evaluate candidate resumes against the given requirements. "
                                "Rules: "
                                "1. A candidate must meet ALL requirements to be Accepted. "
                                "2. If even one requirement is not met, the verdict must be Reject. "
                                "3. List strengths ONLY if they are aligned with the requirements. "
                                "4. List weaknesses ONLY if they are violations or missing requirements. "
                                "5. Always end with a clear verdict: Accept or Reject (no in-between). Clearly state the word accepted or rejected as well. "
                                "Make no assumptions. Consider fail if requirement related detail is not present or sounds ambiguous. "
                                "Ensure, verdict is the LAST line."
                            ),
                        },
                        {"role": "user", "content": f"Job requirements:\n{requirements}"},
                        {"role": "user", "content": f"Candidate resume:\n{text}"},
                    ],
                    stream=False
                )

                results.append({
                    "file": filename,
                    "result": response["message"]["content"]
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)