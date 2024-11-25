// test.rs
use concrete_quantizer::Quantizer;
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;
use std::error::Error;
use std::fs;

#[test]
fn test_quantization() -> Result<(), Box<dyn Error>> {
    println!("Starting test...");

    // Load quantizers
    let input_quantizer = Quantizer::from_json_file("./scripts/input_quantizer.json")?;
    println!("Successfully loaded input quantizer");
    let output_quantizer = Quantizer::from_json_file("./scripts/output_quantizer.json")?;
    println!("Successfully loaded output quantizer");

    // Load test values
    let test_values_data = fs::read_to_string("./scripts/test_values.json")?;
    let test_values_json: Value = serde_json::from_str(&test_values_data)?;
    println!("Successfully loaded test values");

    // Process input values
    if let Some(input_values) = test_values_json.get("input_values") {
        println!("Processing input values...");
        let input_values_list: Vec<Vec<f64>> = serde_json::from_value(input_values.clone())?;
        let num_samples = input_values_list.len();
        let num_features = input_values_list[0].len();
        println!("Input shape: {}x{}", num_samples, num_features);

        let input_values = ArrayD::from_shape_vec(
            IxDyn(&[num_samples, num_features]),
            input_values_list.into_iter().flatten().collect(),
        )?;

        // Quantize input values
        let quantized_input = input_quantizer.quantize(&input_values);
        println!(
            "Quantized input shape: {:?}",
            quantized_input.shape()
        );
    }

    // Process expected dequantized predictions
    let float_predictions_array = if let Some(float_predictions) = test_values_json.get("dequantized_predictions") {
        println!("Processing float predictions...");
        let float_predictions_list: Vec<Vec<f64>> =
            serde_json::from_value(float_predictions.clone())?;
        let num_samples = float_predictions_list.len();
        let num_classes = float_predictions_list[0].len();
        println!(
            "Float predictions shape: {}x{}",
            num_samples, num_classes
        );

        Some(ArrayD::from_shape_vec(
            IxDyn(&[num_samples, num_classes]),
            float_predictions_list.into_iter().flatten().collect(),
        )?)
    } else {
        None
    };

    // Process expectedquantized predictions
    if let Some(quantized_predictions) = test_values_json.get("quantized_predictions") {
        println!("Processing quantized predictions...");
        let quantized_predictions_list: Vec<Vec<i64>> =
            serde_json::from_value(quantized_predictions.clone())?;
        let num_samples = quantized_predictions_list.len();
        let num_classes = quantized_predictions_list[0].len();
        println!(
            "Quantized predictions shape: {}x{}",
            num_samples, num_classes
        );

        let quantized_predictions = ArrayD::from_shape_vec(
            IxDyn(&[num_samples, num_classes]),
            quantized_predictions_list.into_iter().flatten().collect(),
        )?;

        // Dequantize predictions
        let dequantized_predictions =
            output_quantizer.dequantize(&quantized_predictions);
        println!(
            "Dequantized predictions shape: {:?}",
            dequantized_predictions.shape()
        );

        // Compare with float predictions
        if let Some(float_predictions) = float_predictions_array {
            // Compare values with tolerance
            for (dequant_val, float_val) in dequantized_predictions
                .iter()
                .zip(float_predictions.iter())
            {
                assert!(
                    (dequant_val - float_val).abs() < 1e-3,
                    "Mismatch in predictions: Dequantized = {}, Float = {}",
                    dequant_val,
                    float_val
                );
            }
            println!(
                "Successfully verified dequantized predictions match float predictions"
            );
        }
    }

    Ok(())
}