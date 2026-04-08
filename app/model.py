import io
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

# For ImageNet classes, dogs are typically indices 151 through 268
# Cats are indices 281 through 285
def is_dog(class_id):
    return 151 <= class_id <= 268

def is_cat(class_id):
    return 281 <= class_id <= 285

# Load model using the modern syntax
# Using DEFAULT weights ensures we get the most accurate weights available
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()  # Put model into inference mode

# Preprocessing defined by the model weights
preprocess = weights.transforms()

def predict_image(image_bytes: bytes) -> dict:
    """Predicts if an uploaded image is a Cat, Dog, or Other, and returns confidence score."""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess and add batch dimension (N, C, H, W)
        batch = preprocess(image).unsqueeze(0)

        # Disable gradient calculations for inference speed
        with torch.no_grad():
            output = model(batch)
            # Convert final output logits to probabilities
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the top class id  
        class_id = probabilities.argmax().item()
        confidence = probabilities[class_id].item()

        if is_cat(class_id):
            return {"prediction": "Cat", "confidence": confidence}
        elif is_dog(class_id):
            return {"prediction": "Dog", "confidence": confidence}
        else:
            return {"prediction": "Other", "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}
