use concrete_quantizer::Quantizer;
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;
use std::error::Error;
use std::fs;

#[test]
fn test_quantization() -> Result<(), Box<dyn Error>> {
    // Paths to the JSON files generated by the Python script
    let quantizer_json_path = "scripts/quantizer.json";
    let test_values_json_path = "scripts/test_values.json";

    // Initialize the Quantizer from the JSON file
    let quantizer = Quantizer::from_json_file(quantizer_json_path)?;

    // Read the test values from JSON
    let test_values_data = fs::read_to_string(test_values_json_path)?;
    let test_values_json: Value = serde_json::from_str(&test_values_data)?;

    // Read the original values from JSON
    let values_list: Vec<f64> = serde_json::from_value(
        test_values_json["values"].clone(),
    ).unwrap();
    let values = ArrayD::from_shape_vec(IxDyn(&[values_list.len()]), values_list).unwrap();

    // Quantize the values in Rust
    let qvalues_rust = quantizer.quantize(&values);

    // Dequantize the quantized values in Rust
    let dequantized_values_rust = quantizer.dequantize(&qvalues_rust);

    // Read the quantized values from JSON
    let qvalues_python_list: Vec<i64> = serde_json::from_value(
        test_values_json["qvalues"].clone(),
    ).unwrap();
    let qvalues_python = ArrayD::from_shape_vec(
        IxDyn(&[qvalues_python_list.len()]),
        qvalues_python_list,
    ).unwrap();

    // Read the dequantized values from JSON
    let dequantized_values_python_list: Vec<f64> = serde_json::from_value(
        test_values_json["dequantized_values"].clone(),
    ).unwrap();
    let dequantized_values_python = ArrayD::from_shape_vec(
        IxDyn(&[dequantized_values_python_list.len()]),
        dequantized_values_python_list,
    ).unwrap();

    // Compare the quantized values
    assert_eq!(qvalues_rust, qvalues_python);

    // Compare the dequantized values within a tolerance
    for (rust_val, python_val) in dequantized_values_rust
        .iter()
        .zip(dequantized_values_python.iter())
    {
        assert!(
            (rust_val - python_val).abs() < 1e-6,
            "Mismatch in dequantized values: Rust = {}, Python = {}",
            rust_val,
            python_val
        );
    }

    Ok(())
}