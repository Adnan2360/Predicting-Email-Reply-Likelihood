
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def init():
    global model, tokenizer, device

    # Azure gives the registered model location
    # The actual Hugging Face files are inside this subfolder
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "distilbert_email_reply_model"
    )

    # Load the fine-tuned tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    device = torch.device("cpu")
    model.to(device)
    model.eval()


def run(raw_data):
    # Read incoming email text
    data = json.loads(raw_data)
    email_text = data["email_text"]

    # Convert email text into DistilBERT tokens
    inputs = tokenizer(
        email_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    reply_probability = probabilities[0][1].item()
    predicted_class = int(
        torch.argmax(probabilities, dim=1).item()
    )

    prediction = (
        "Reply"
        if predicted_class == 1
        else "No Reply"
    )

    return {
        "prediction": prediction,
        "reply_probability": round(reply_probability, 4)
    }
