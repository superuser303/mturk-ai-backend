# app.py - Free Cloud MTurk AI API
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import clip, torch, io
from PIL import Image
import base64
from io import BytesIO

app = FastAPI()
device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to(device)

@app.post("/analyze")
async def analyze(task_type: str, prompt: str = None, images_b64: list = File(...)):
    images = []
    for b64 in images_b64:
        img_data = base64.b64decode(b64.split(",")[1])
        images.append(Image.open(BytesIO(img_data)).convert("RGB"))
    
    if task_type == "image_eval":
        scores = []
        for img in images:
            text = clip.tokenize([prompt]).to(device)
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                scores.append(clip_model(img_t, text)[0].item())
        return {"decision": scores.index(max(scores))}
    
    elif task_type == "style_annot":
        img1, img2 = images
        def feat(img):
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad(): return resnet(x)
        sim = torch.cosine_similarity(feat(img1), feat(img2)).item()
        label = "Hell Yes" if sim >= 0.95 else "Yes" if sim >= 0.85 else "No" if sim >= 0.60 else "Hell No"
        return {"decision": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)