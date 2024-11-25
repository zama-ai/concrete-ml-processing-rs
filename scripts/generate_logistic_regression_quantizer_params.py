from concrete.ml.sklearn import LogisticRegression
from sklearn.datasets import load_iris
import json
import numpy as np

# Load data
X_train, y_train = load_iris(return_X_y=True)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save input quantization parameters
with open("input_quantizer.json", "w") as f:
    for quantizer in model.input_quantizers:
        quantizer.dump(f)

# Save output quantization parameters
with open("output_quantizer.json", "w") as f:
    for quantizer in model.output_quantizers:
        quantizer.dump(f)

# Generate test values and predictions
float_test_samples = X_train[0:5]  # Taking first 5 samples as test data

# Quantize input
quantized_test_samples = model.quantize_input(float_test_samples)

# Prediction on quantized input
quantized_predictions = model._inference(quantized_test_samples)

# Dequantize output
dequantized_predictions = model.dequantize_output(quantized_predictions)

test_data = {
    "input_fp32_values": quantized_test_samples.tolist(),
    "input_quantized_values": quantized_test_samples.tolist(),
    "quantized_predictions": quantized_predictions.tolist(),
    "dequantized_predictions": dequantized_predictions.tolist()
}

# Save test values and predictions
with open("test_values.json", "w") as f:
    json.dump(test_data, f, indent=2)
