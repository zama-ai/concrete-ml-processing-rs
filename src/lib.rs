use serde::{Deserialize, Serialize};
use ndarray::ArrayD;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedFloat {
    pub type_name: String,
    pub serialized_value: f64,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedArray {
    pub type_name: String,
    pub serialized_value: Vec<f64>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizerSerializedValue {
    pub n_bits: u32,
    pub is_signed: bool,
    pub is_symmetric: bool,
    pub is_qat: bool,
    pub is_narrow: bool,
    pub is_precomputed_qat: bool,
    pub rmax: SerializedFloat,
    pub rmin: SerializedFloat,
    pub uvalues: SerializedArray,
    pub scale: SerializedFloat,
    pub zero_point: i64,
    pub offset: i64,
    pub no_clipping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformQuantizerSerialized {
    pub type_name: String,
    pub serialized_value: QuantizerSerializedValue,
}

#[derive(Debug, Clone)]
pub struct Quantizer {
    pub n_bits: u32,
    pub is_signed: bool,
    pub is_symmetric: bool,
    pub is_qat: bool,
    pub is_narrow: bool,
    pub is_precomputed_qat: bool,
    pub rmax: f64,
    pub rmin: f64,
    pub uvalues: Vec<f64>,
    pub scale: f64,
    pub zero_point: i64,
    pub offset: i64,
    pub no_clipping: bool,
}

impl Quantizer {
    /// Initialize the Quantizer from a JSON string.
    pub fn from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        // Deserialize the top-level structure
        let uq_serialized: UniformQuantizerSerialized = serde_json::from_str(json_str)?;

        // Extract and convert the nested serialized values
        let serialized_value = uq_serialized.serialized_value;

        Ok(Quantizer {
            n_bits: serialized_value.n_bits,
            is_signed: serialized_value.is_signed,
            is_symmetric: serialized_value.is_symmetric,
            is_qat: serialized_value.is_qat,
            is_narrow: serialized_value.is_narrow,
            is_precomputed_qat: serialized_value.is_precomputed_qat,
            rmax: serialized_value.rmax.serialized_value,
            rmin: serialized_value.rmin.serialized_value,
            uvalues: serialized_value.uvalues.serialized_value,
            scale: serialized_value.scale.serialized_value,
            zero_point: serialized_value.zero_point,
            offset: serialized_value.offset,
            no_clipping: serialized_value.no_clipping,
        })
    }

    /// Initialize the Quantizer from a JSON file.
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let params_data = std::fs::read_to_string(path)?;
        let quantizer = Quantizer::from_json_str(&params_data)?;
        Ok(quantizer)
    }

    /// Quantize the given floating-point values.
    pub fn quantize(&self, values: &ArrayD<f64>) -> ArrayD<i64> {
        let qvalues = values.mapv(|x| {
            if x.is_nan() {
                // Handle NaN values if necessary
                0
            } else {
                ((x / self.scale + self.zero_point as f64).round()) as i64
            }
        });

        if (!self.is_qat || self.is_precomputed_qat) && !self.no_clipping {
            let mut min_value = if self.is_signed { -self.offset } else { 0 };

            if self.is_narrow {
                min_value += 1;
            }

            let max_value = (1 << self.n_bits) - 1 - self.offset;

            qvalues.mapv(|x| x.clamp(min_value, max_value))
        } else {
            qvalues
        }
    }

    /// Dequantize the given quantized integer values.
    pub fn dequantize(&self, qvalues: &ArrayD<i64>) -> ArrayD<f64> {
        qvalues.mapv(|q| self.scale * ((q as f64) - self.zero_point as f64))
    }
}
