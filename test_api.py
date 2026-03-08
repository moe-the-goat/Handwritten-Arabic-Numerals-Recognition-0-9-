"""Quick API test: send real training images through the predict endpoint."""
import urllib.request, json, base64, glob

N_PER_CLASS = 20
correct = 0
total = 0

for d in range(10):
    imgs = sorted(glob.glob(f"Handwritten Arabic Numerals (0-9)/ANGKA ARAB/{d}/*.png"))
    class_correct = 0
    for i in range(min(N_PER_CLASS, len(imgs))):
        with open(imgs[i], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = json.dumps({"image": f"data:image/png;base64,{b64}"}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8001/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        pred = data["predicted_digit"]
        if pred == d:
            class_correct += 1
        total += 1
    correct += class_correct
    print(f"Digit {d}: {class_correct}/{N_PER_CLASS} correct")

print(f"\nOverall: {correct}/{total} = {correct/total*100:.1f}%")
