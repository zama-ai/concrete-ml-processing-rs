from concrete.ml.quantization.quantizers import (
    QuantizationOptions,
    MinMaxQuantizationStats,
    UniformQuantizer,
    QuantizedArray
)
import numpy as np
import json
import os

def main():
    # Generate more complex sample data
    values = np.linspace(-1.0, 1.0, num=100)

    # Advanced Quantization setup with custom options
    options = QuantizationOptions(
        n_bits=8,
        is_signed=True,
        is_symmetric=True,
        is_qat=False
    )

    # Compute quantization statistics from the values
    stats = MinMaxQuantizationStats()
    stats.compute_quantization_stats(values)

    # Initialize the quantizer with the options and stats
    quantizer = UniformQuantizer(options, stats)
    quantizer.compute_quantization_parameters(options, stats)

    # Create a QuantizedArray using the quantizer
    q_array = QuantizedArray(
        n_bits=options.n_bits,
        values=values,
        value_is_float=True,
        options=options,
        stats=stats
    )

    # Retrieve quantized values
    qvalues = q_array.qvalues

    # Dequantize the quantized values to get the actual dequantized values
    dequantized_values = quantizer.dequant(qvalues)

    # Save the quantizer using the dump method
    quantizer_path = os.path.join(os.path.dirname(__file__), 'quantizer.json')
    with open(quantizer_path, 'w') as f:
        quantizer.dump(f)

    # Save quantized and dequantized values to a separate JSON file for tests
    test_values = {
        'values': values.tolist(),
        'qvalues': qvalues.tolist(),
        'dequantized_values': dequantized_values.tolist()
    }

    test_values_path = os.path.join(os.path.dirname(__file__), 'test_values.json')
    with open(test_values_path, 'w') as f:
        json.dump(test_values, f)

if __name__ == '__main__':
    main()
