from concrete.ml.sklearn import LogisticRegression
from sklearn.datasets import load_iris
import json
import numpy as np
from pathlib import Path

# Load data
X_train, y_train = load_iris(return_X_y=True)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Extract quantized model parameters
q_weights = model._q_weights
q_bias = model._q_bias

# Get script directory
script_dir = Path(__file__).parent

# Save weights and bias
(script_dir / "q_weights.json").write_text(
    json.dumps(q_weights.tolist())
)

(script_dir / "q_bias.json").write_text(
    json.dumps(q_bias.tolist())
)

# Save input quantization parameters
with (script_dir / "input_quantizer.json").open("w") as f:
    for quantizer in model.input_quantizers:
        quantizer.dump(f)

# Save output quantization parameters
with (script_dir / "output_quantizer.json").open("w") as f:
    for quantizer in model.output_quantizers:
        quantizer.dump(f)

# Save weights quantization parameters
with (script_dir / "weights_quantizer.json").open("w") as f:
    f.write(
        json.dumps(model._weight_quantizer.dump(f))
)

# Generate test values and predictions
float_test_samples = X_train[0:5]  # Taking first 5 samples as test data

# Quantize input
quantized_test_samples = model.quantize_input(float_test_samples)

# Prediction on quantized input
quantized_predictions = model._inference(quantized_test_samples)

# Dequantize output
dequantized_predictions = model.dequantize_output(quantized_predictions)

test_data = {
    "input_fp32_values": float_test_samples.tolist(),
    "input_quantized_values": quantized_test_samples.tolist(),
    "quantized_predictions": quantized_predictions.tolist(),
    "dequantized_predictions": dequantized_predictions.tolist()
}

# Save test values and predictions
(script_dir / "test_values.json").write_text(
    json.dumps(test_data, indent=2)
)
