mod bit_pack_array;
mod fsst_array;

use std::sync::Arc;

use arrow_array::builder::StringDictionaryBuilder;
use arrow_array::{
    cast::AsArray,
    types::{Int32Type, UInt32Type},
    DictionaryArray, StringArray,
};
use arrow_array::{PrimitiveArray, RecordBatch};

use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field, Schema};
use bit_pack_array::BitPackedArray;
use fsst::Compressor;
use fsst_array::FsstArray;

#[derive(Debug)]
pub struct EtcStringMetadata {
    compressor: Arc<Compressor>,
    uncompressed_len: u32,
    keys_original_len: u32,
    keys_bit_width: u8,
}

pub struct EtcStringArray {
    keys: BitPackedArray<UInt32Type>,
    values: FsstArray,
}

impl EtcStringArray {
    pub fn from_string_array(array: &StringArray, compressor: Option<Arc<Compressor>>) -> Self {
        let dict = string_to_dict_string(array);
        let (keys, values) = dict.into_parts();

        let (_, keys, nulls) = keys.into_parts();
        let keys: ScalarBuffer<u32> = unsafe { std::mem::transmute(keys) };
        let keys = PrimitiveArray::<UInt32Type>::try_new(keys, nulls).unwrap();

        let distinct_count = values.len();
        let max_bit_width = ceil_log2(distinct_count as u64);
        debug_assert!(2u64.pow(max_bit_width as u32) >= distinct_count as u64);

        let bit_packed_array = BitPackedArray::from_primitive(keys, max_bit_width as u8);

        let dict_values = values.as_string::<i32>();

        let fsst_values = match compressor {
            Some(compressor) => {
                FsstArray::from_string_array_with_compressor(dict_values, compressor)
            }
            None => FsstArray::from(dict_values),
        };
        EtcStringArray {
            keys: bit_packed_array,
            values: fsst_values,
        }
    }

    pub fn compressor(&self) -> Arc<Compressor> {
        self.values.compressor.clone()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn to_dict_string(&self) -> DictionaryArray<UInt32Type> {
        let primitive_key = self.keys.to_primitive().clone();
        let values: StringArray = StringArray::from(&self.values);
        let dict = unsafe {
            DictionaryArray::<UInt32Type>::new_unchecked(primitive_key, Arc::new(values))
        };
        dict
    }

    pub fn to_string_array(&self) -> StringArray {
        let dict = self.to_dict_string();
        let value = arrow_cast::cast(&dict, &DataType::Utf8).unwrap();
        value.as_string::<i32>().clone()
    }

    /// Repackage the data into Arrow-compatible format, so that it can be written to disk, transferred over flight.
    pub fn to_record_batch(&self) -> (RecordBatch, EtcStringMetadata) {
        let schema = Schema::new(vec![
            Field::new("keys", DataType::UInt32, false),
            Field::new("values", DataType::Binary, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(self.keys.values.clone()),
                Arc::new(self.values.compressed.clone()),
            ],
        )
        .unwrap();
        (batch, self.metadata())
    }

    pub fn from_record_batch(batch: RecordBatch, metadata: &EtcStringMetadata) -> Self {
        let key_column = batch.column(0).as_primitive::<UInt32Type>();
        let values_column = batch.column(1).as_binary();

        let keys = BitPackedArray::from_parts(
            key_column.clone(),
            metadata.keys_bit_width,
            metadata.keys_original_len as usize,
        );
        let values = FsstArray::from_parts(
            values_column.clone(),
            metadata.compressor.clone(),
            metadata.uncompressed_len as usize,
        );
        EtcStringArray { keys, values }
    }

    pub fn metadata(&self) -> EtcStringMetadata {
        EtcStringMetadata {
            compressor: self.values.compressor.clone(),
            uncompressed_len: self.values.uncompressed_len as u32,
            keys_original_len: self.keys.values.len() as u32,
            keys_bit_width: self.keys.bit_width,
        }
    }

    pub fn get_array_memory_size(&self) -> usize {
        self.keys.get_array_memory_size() + self.values.get_array_memory_size()
    }
}

fn string_to_dict_string(input: &StringArray) -> DictionaryArray<Int32Type> {
    let mut builder = StringDictionaryBuilder::<Int32Type>::new();
    for s in input.iter() {
        builder.append_option(s);
    }
    builder.finish()
}

fn ceil_log2(n: u64) -> u8 {
    if n <= 1 {
        0
    } else {
        let log = 64 - n.leading_zeros() as u8 - 1;
        if n.is_power_of_two() {
            log
        } else {
            log + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    #[test]
    fn test_simple_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let etc = EtcStringArray::from_string_array(&input, None);
        let output = etc.to_string_array();

        assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), output.value(i));
        }
    }

    #[test]
    fn test_roundtrip_with_nulls() {
        let input = StringArray::from(vec![
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("hello"),
        ]);
        let etc = EtcStringArray::from_string_array(&input, None);
        let output = etc.to_string_array();

        assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            if input.is_null(i) {
                assert!(output.is_null(i));
            } else {
                assert_eq!(input.value(i), output.value(i));
            }
        }
    }

    #[test]
    fn test_roundtrip_with_many_duplicates() {
        let values = vec!["a", "b", "c"];
        let input: Vec<&str> = (0..1000).map(|i| values[i % values.len()]).collect();
        let input = StringArray::from(input);

        let etc = EtcStringArray::from_string_array(&input, None);
        let output = etc.to_string_array();

        assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), output.value(i));
        }
    }

    #[test]
    fn test_roundtrip_with_long_strings() {
        let input = StringArray::from(vec![
            "This is a very long string that should be compressed well",
            "Another long string with some common patterns",
            "This is a very long string that should be compressed well",
            "Some unique text here to mix things up",
            "Another long string with some common patterns",
        ]);

        let etc = EtcStringArray::from_string_array(&input, None);
        let output = etc.to_string_array();

        assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), output.value(i));
        }
    }

    #[test]
    fn test_empty_strings() {
        let input = StringArray::from(vec!["", "", "non-empty", ""]);
        let etc = EtcStringArray::from_string_array(&input, None);
        let output = etc.to_string_array();

        assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), output.value(i));
        }
    }

    #[test]
    fn test_dictionary_roundtrip() {
        let input = StringArray::from(vec!["hello", "world", "hello", "rust"]);
        let etc = EtcStringArray::from_string_array(&input, None);
        let dict = etc.to_dict_string();

        // Check dictionary values are unique
        let dict_values = dict.values();
        let unique_values: std::collections::HashSet<&str> = dict_values
            .as_string::<i32>()
            .into_iter()
            .flatten()
            .collect();

        assert_eq!(unique_values.len(), 3); // "hello", "world", "rust"

        // Convert back to string array and verify
        let output = etc.to_string_array();
        assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            assert_eq!(input.value(i), output.value(i));
        }
    }
}
