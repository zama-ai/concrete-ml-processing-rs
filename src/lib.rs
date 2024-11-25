use serde::{Deserialize, Serialize};
use ndarray::{ArrayD, IxDyn, Zip};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedValue<T> {
    pub type_name: String,
    pub serialized_value: T,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OffsetValue {
    Direct(i64),
    Array(SerializedValue<Vec<i64>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ZeroPointValue {
    Direct(SerializedValue<i64>),
    Array(SerializedValue<Vec<i64>>),
    NestedArray(SerializedValue<Vec<Vec<i64>>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizerSerializedValue {
    #[serde(default = "default_n_bits")]
    pub n_bits: u32,
    pub is_signed: bool,
    pub is_symmetric: bool,
    pub is_qat: bool,
    pub is_narrow: bool,
    pub is_precomputed_qat: bool,
    pub rmax: Option<SerializedValue<f64>>,
    pub rmin: Option<SerializedValue<f64>>,
    pub uvalues: Option<SerializedValue<Vec<f64>>>,
    pub scale: SerializedValue<f64>,
    pub zero_point: ZeroPointValue,
    pub offset: OffsetValue,
    pub no_clipping: bool,
}

fn default_n_bits() -> u32 {
    8
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
    pub zero_point: ArrayD<f64>,
    pub offset: ArrayD<f64>,
    pub no_clipping: bool,
}

impl Quantizer {
    /// Initialize the Quantizer from a JSON string.
    pub fn from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        // Deserialize the top-level structure
        let uq_serialized: UniformQuantizerSerialized = serde_json::from_str(json_str)?;
        let serialized_value = uq_serialized.serialized_value;

        // Parse zero_point and offset
        let zero_point = parse_zero_point_value(serialized_value.zero_point);
        let offset = parse_offset_value(serialized_value.offset);

        Ok(Quantizer {
            n_bits: serialized_value.n_bits,
            is_signed: serialized_value.is_signed,
            is_symmetric: serialized_value.is_symmetric,
            is_qat: serialized_value.is_qat,
            is_narrow: serialized_value.is_narrow,
            is_precomputed_qat: serialized_value.is_precomputed_qat,
            rmax: serialized_value
                .rmax
                .map_or(0.0, |v| v.serialized_value),
            rmin: serialized_value
                .rmin
                .map_or(0.0, |v| v.serialized_value),
            uvalues: serialized_value
                .uvalues
                .map_or(Vec::new(), |v| v.serialized_value),
            scale: serialized_value.scale.serialized_value,
            zero_point,
            offset,
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
        // Broadcast zero_point and offset to the shape of values
        let zero_point = self.zero_point.broadcast(values.shape()).unwrap();
        let offset = self.offset.broadcast(values.shape()).unwrap();

        // Quantize
        let mut qvalues = (values / self.scale + &zero_point).mapv(|x| x.round() as i64);

        if (!self.is_qat || self.is_precomputed_qat) && !self.no_clipping {
            // Calculate min and max values for clamping
            let mut min_value = if self.is_signed {
                offset.mapv(|o| -o as i64)
            } else {
                ArrayD::zeros(qvalues.raw_dim())
            };

            if self.is_narrow {
                min_value += 1;
            }

            let max_value = ArrayD::from_elem(
                qvalues.raw_dim(),
                ((1 << self.n_bits) - 1) as i64,
            ) - &offset.mapv(|o| o as i64);

            // Clamp qvalues between min_value and max_value
            Zip::from(&mut qvalues)
                .and(&min_value)
                .and(&max_value)
                .for_each(|x, &min, &max| {
                    *x = (*x).clamp(min, max);
                });
        }

        qvalues
    }

    /// Dequantize the given quantized integer values.
    pub fn dequantize(&self, qvalues: &ArrayD<i64>) -> ArrayD<f64> {
        // Broadcast zero_point to the shape of qvalues
        let zero_point = self.zero_point.broadcast(qvalues.shape()).unwrap();
        let qvalues_f64 = qvalues.mapv(|q| q as f64);
        (&qvalues_f64 - &zero_point) * self.scale
    }
}

/// Helper function to parse OffsetValue into ArrayD<f64>
fn parse_offset_value(value: OffsetValue) -> ArrayD<f64> {
    match value {
        OffsetValue::Direct(v) => ArrayD::from_elem(IxDyn(&[1]), v as f64),
        OffsetValue::Array(array) => {
            let data = array
                .serialized_value
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap()
        }
    }
}

/// Helper function to parse ZeroPointValue into ArrayD<f64>
fn parse_zero_point_value(value: ZeroPointValue) -> ArrayD<f64> {
    match value {
        ZeroPointValue::Direct(serialized) => {
            ArrayD::from_elem(IxDyn(&[1]), serialized.serialized_value as f64)
        }
        ZeroPointValue::Array(serialized) => {
            let data = serialized
                .serialized_value
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap()
        }
        ZeroPointValue::NestedArray(serialized) => {
            let shape = vec![
                serialized.serialized_value.len(),
                serialized.serialized_value[0].len(),
            ];
            let data = serialized
                .serialized_value
                .into_iter()
                .flatten()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
        }
    }
}