FROM python:3.10-slim

WORKDIR /code

# Pre-install dependencies to cache
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy source code
COPY ./app /code/app

# Download model weights into Docker image
# This avoids downloading the weights on the first request
RUN python -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
