/// SafeTensors format parser.
///
/// SafeTensors is a simple, safe binary format for storing tensors:
///
/// ```text
/// [8 bytes: header_size as u64 LE]
/// [header_size bytes: JSON metadata]
/// [remaining bytes: raw tensor data]
/// ```
///
/// The JSON metadata maps tensor names to their dtype, shape, and byte offsets
/// within the data section. This parser extracts f32 tensors from the file.
///
/// No external SafeTensors crate is used — the format is simple enough
/// to parse directly.

use std::collections::HashMap;
use std::path::Path;
use serde::Deserialize;
use crate::error::{HypEmbedError, Result};
use crate::tensor::{Tensor, Shape};

/// Metadata for a single tensor in a SafeTensors file.
#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    /// Data type string (e.g., "F32", "F16", "BF16").
    pub dtype: String,
    /// Shape as a list of dimension sizes.
    pub shape: Vec<usize>,
    /// Byte offsets [start, end) within the data section.
    pub data_offsets: [usize; 2],
}

/// A parsed SafeTensors file.
#[derive(Debug)]
pub struct SafeTensorsFile {
    /// Tensor metadata by name.
    pub tensors: HashMap<String, TensorInfo>,
    /// Raw data section (after the header).
    pub data: Vec<u8>,
}

impl SafeTensorsFile {
    /// Load and parse a SafeTensors file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;

        if bytes.len() < 8 {
            return Err(HypEmbedError::Model("SafeTensors file too small".into()));
        }

        // Read header size (8 bytes, little-endian u64)
        let header_size = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize;

        let header_end = 8 + header_size;
        if header_end > bytes.len() {
            return Err(HypEmbedError::Model(format!(
                "SafeTensors header extends beyond file: header_end={}, file_size={}",
                header_end,
                bytes.len()
            )));
        }

        // Parse JSON header
        let header_json = std::str::from_utf8(&bytes[8..header_end]).map_err(|e| {
            HypEmbedError::Model(format!("Invalid UTF-8 in SafeTensors header: {}", e))
        })?;

        // The header is a JSON object mapping tensor names to TensorInfo.
        // There may also be a "__metadata__" key that we ignore.
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;

        let mut tensors = HashMap::new();
        for (name, value) in &raw {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(value.clone()).map_err(|e| {
                HypEmbedError::Model(format!(
                    "Failed to parse tensor info for '{}': {}",
                    name, e
                ))
            })?;
            tensors.insert(name.clone(), info);
        }

        let data = bytes[header_end..].to_vec();

        Ok(SafeTensorsFile { tensors, data })
    }

    /// Extract a tensor by name as f32.
    ///
    /// Currently supports F32 and F16 dtypes. F16 is converted to F32.
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| {
            HypEmbedError::Model(format!("Tensor '{}' not found in SafeTensors file", name))
        })?;

        let start = info.data_offsets[0];
        let end = info.data_offsets[1];

        if end > self.data.len() {
            return Err(HypEmbedError::Model(format!(
                "Tensor '{}' data offsets [{}, {}) exceed data section size {}",
                name, start, end, self.data.len()
            )));
        }

        let raw = &self.data[start..end];
        let shape = Shape::new(info.shape.clone());

        match info.dtype.as_str() {
            "F32" => {
                let expected_bytes = shape.numel() * 4;
                if raw.len() != expected_bytes {
                    return Err(HypEmbedError::Model(format!(
                        "Tensor '{}': expected {} bytes for F32 shape {:?}, got {}",
                        name, expected_bytes, info.shape, raw.len()
                    )));
                }
                let floats: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(floats, shape)
            }
            "F16" => {
                let expected_bytes = shape.numel() * 2;
                if raw.len() != expected_bytes {
                    return Err(HypEmbedError::Model(format!(
                        "Tensor '{}': expected {} bytes for F16 shape {:?}, got {}",
                        name, expected_bytes, info.shape, raw.len()
                    )));
                }
                // Convert F16 to F32
                let floats: Vec<f32> = raw
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits)
                    })
                    .collect();
                Tensor::from_vec(floats, shape)
            }
            other => Err(HypEmbedError::Model(format!(
                "Unsupported dtype '{}' for tensor '{}'",
                other, name
            ))),
        }
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

/// Convert an IEEE 754 half-precision (f16) value to f32.
///
/// This implements the standard conversion:
/// - Sign: bit 15
/// - Exponent: bits 14-10 (5 bits, biased by 15)
/// - Mantissa: bits 9-0 (10 bits)
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            return f32::from_bits(sign << 31);
        }
        // Subnormal: (−1)^sign × 2^(−14) × (0.mantissa)
        // Convert to f32 by normalizing
        let mut m = mant;
        let mut e = -14i32 + 127; // f32 bias
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // Remove leading 1
        let f32_bits = (sign << 31) | ((e as u32) << 23) | (m << 13);
        return f32::from_bits(f32_bits);
    }

    if exp == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        return f32::from_bits(f32_bits);
    }

    // Normal: adjust exponent from f16 bias (15) to f32 bias (127)
    let f32_exp = exp + 127 - 15;
    let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16 representation of 1.0: sign=0, exp=15 (0b01111), mant=0
        // bits = 0_01111_0000000000 = 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_neg_two() {
        // f16 representation of -2.0: sign=1, exp=16 (0b10000), mant=0
        // bits = 1_10000_0000000000 = 0xC000
        let val = f16_to_f32(0xC000);
        assert!((val - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // f16 representation of 0.5: sign=0, exp=14 (0b01110), mant=0
        // bits = 0_01110_0000000000 = 0x3800
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5).abs() < 1e-6);
    }
}
