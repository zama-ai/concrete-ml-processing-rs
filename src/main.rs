use concrete_quantizer::Quantizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to your JSON file
    let json_path = "scripts/quantizer.json";

    // Initialize the Quantizer from the JSON file
    let quantizer = Quantizer::from_json_file(json_path)?;

    // Example floating-point values to quantize
    let values = ndarray::Array::linspace(-1.0, 1.0, 100).into_dyn();

    // Quantize the values
    let qvalues = quantizer.quantize(&values);

    // Dequantize the values
    let dequantized_values = quantizer.dequantize(&qvalues);

    // Print the results
    println!("Original Values: {:?}", values);
    println!("Quantized Values: {:?}", qvalues);
    println!("Dequantized Values: {:?}", dequantized_values);

    Ok(())
}
