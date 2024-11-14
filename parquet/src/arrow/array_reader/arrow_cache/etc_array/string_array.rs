use std::any::Any;
use std::num::NonZero;
use std::sync::Arc;

use arrow_array::builder::StringDictionaryBuilder;
use arrow_array::{cast::AsArray, types::UInt32Type, DictionaryArray, StringArray};
use arrow_array::{Array, ArrayRef, BinaryArray, BooleanArray, PrimitiveArray, RecordBatch};

use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field, Schema};
use fsst::Compressor;

use crate::arrow::arrow_cache::etc_array::{get_bit_width, FsstArray};

use super::{BitPackedArray, EtcArray, EtcArrayRef};

impl EtcArray for EtcStringArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_array_memory_size(&self) -> usize {
        self.keys.get_array_memory_size() + self.values.get_array_memory_size()
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn to_arrow_array(&self) -> (ArrayRef, Schema) {
        let dict = self.to_string_array();
        let schema = Schema::new(vec![Field::new("", DataType::Utf8, false)]);
        (Arc::new(dict), schema)
    }

    fn filter(&self, selection: &BooleanArray) -> EtcArrayRef {
        let values = self.values.clone();
        let keys = self.keys.clone();
        let primitive_keys = keys.to_primitive();
        let filtered_keys = arrow_select::filter::filter(&primitive_keys, selection)
            .unwrap()
            .as_primitive::<UInt32Type>()
            .clone();
        let bit_packed_array = BitPackedArray::from_primitive(filtered_keys, keys.bit_width);
        Arc::new(EtcStringArray {
            keys: bit_packed_array,
            values,
        })
    }
}

/// Metadata for the EtcStringArray.
#[derive(Debug)]
pub struct EtcStringMetadata {
    compressor: Arc<Compressor>,
    uncompressed_len: u32,
    keys_original_len: u32,
    keys_bit_width: NonZero<u8>,
}

/// An array that stores strings in a dictionary format, with a bit-packed array for the keys and a FSST array for the values.
#[derive(Debug)]
pub struct EtcStringArray {
    keys: BitPackedArray<UInt32Type>,
    values: FsstArray,
}

impl EtcStringArray {
    /// Create an EtcStringArray from a StringArray.
    pub fn from_string_array(array: &StringArray, compressor: Option<Arc<Compressor>>) -> Self {
        let dict = string_to_dict_string(array);
        let (keys, values) = dict.into_parts();

        let (_, keys, nulls) = keys.into_parts();
        let keys = PrimitiveArray::<UInt32Type>::try_new(keys, nulls).unwrap();

        let distinct_count = values.len();
        let max_bit_width = get_bit_width(distinct_count as u64);
        debug_assert!(2u64.pow(max_bit_width.get() as u32) >= distinct_count as u64);

        let bit_packed_array =
            BitPackedArray::from_primitive(keys, max_bit_width);

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

    /// Get the compressor of the EtcStringArray.
    pub fn compressor(&self) -> Arc<Compressor> {
        self.values.compressor.clone()
    }

    /// Convert the EtcStringArray to a DictionaryArray.
    pub fn to_dict_string(&self) -> DictionaryArray<UInt32Type> {
        let primitive_key = self.keys.to_primitive().clone();
        let values: StringArray = StringArray::from(&self.values);
        unsafe { DictionaryArray::<UInt32Type>::new_unchecked(primitive_key, Arc::new(values)) }
    }

    /// Convert the EtcStringArray to a StringArray.
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

    /// Reconstruct the EtcStringArray from a RecordBatch.
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

    /// Get the metadata of the EtcStringArray.
    pub fn metadata(&self) -> EtcStringMetadata {
        EtcStringMetadata {
            compressor: self.values.compressor.clone(),
            uncompressed_len: self.values.uncompressed_len as u32,
            keys_original_len: self.keys.values.len() as u32,
            keys_bit_width: self.keys.bit_width,
        }
    }

    /// Compare the values of the EtcStringArray with a given string and return a BooleanArray of the result.
    pub fn compare_not_equals(&self, needle: &str) -> BooleanArray {
        let result = self.compare_equals(needle);
        let (values, nulls) = result.into_parts();
        let values = !&values;
        BooleanArray::new(values, nulls)
    }

    /// Compare the values of the EtcStringArray with a given string.
    pub fn compare_equals(&self, needle: &str) -> BooleanArray {
        let compressor = &self.values.compressor;
        let compressed = compressor.compress(needle.as_bytes());
        let compressed = BinaryArray::new_scalar(compressed);

        let values = &self.values.compressed;

        let result = arrow_ord::cmp::eq(values, &compressed).unwrap();
        let (values, nulls) = result.into_parts();
        assert!(
            nulls.is_none(),
            "The dictionary values should not have nulls"
        );

        let keys = self.keys.to_primitive();
        let mut return_mask = BooleanBuffer::new_unset(keys.len());
        for idx in values.set_indices() {
            let result =
                arrow_ord::cmp::eq(&keys, &PrimitiveArray::<UInt32Type>::new_scalar(idx as u32))
                    .unwrap();
            let (values, _nulls) = result.into_parts();
            return_mask = &return_mask | &values;
        }
        BooleanArray::new(return_mask, keys.nulls().cloned())
    }
}

fn string_to_dict_string(input: &StringArray) -> DictionaryArray<UInt32Type> {
    let mut builder = StringDictionaryBuilder::<UInt32Type>::new();
    for s in input.iter() {
        builder.append_option(s);
    }
    builder.finish()
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

    #[test]
    fn test_compare_equals_comprehensive() {
        struct TestCase<'a> {
            input: Vec<Option<&'a str>>,
            needle: &'a str,
            expected: Vec<Option<bool>>,
        }

        let test_cases = vec![
            TestCase {
                input: vec![Some("hello"), Some("world"), Some("hello"), Some("rust")],
                needle: "hello",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("hello"), Some("world"), Some("hello"), Some("rust")],
                needle: "nonexistent",
                expected: vec![Some(false), Some(false), Some(false), Some(false)],
            },
            TestCase {
                input: vec![Some("hello"), None, Some("hello"), None, Some("world")],
                needle: "hello",
                expected: vec![Some(true), None, Some(true), None, Some(false)],
            },
            TestCase {
                input: vec![Some(""), Some("hello"), Some(""), Some("world")],
                needle: "",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("same"), Some("same"), Some("same"), Some("same")],
                needle: "same",
                expected: vec![Some(true), Some(true), Some(true), Some(true)],
            },
            TestCase {
                input: vec![
                    Some("apple"),
                    None,
                    Some("banana"),
                    Some("apple"),
                    None,
                    Some("cherry"),
                ],
                needle: "apple",
                expected: vec![Some(true), None, Some(false), Some(true), None, Some(false)],
            },
            TestCase {
                input: vec![Some("Hello"), Some("hello"), Some("HELLO"), Some("HeLLo")],
                needle: "hello",
                expected: vec![Some(false), Some(true), Some(false), Some(false)],
            },
            TestCase {
                input: vec![
                    Some("こんにちは"), // "Hello" in Japanese
                    Some("世界"),       // "World" in Japanese
                    Some("こんにちは"),
                    Some("rust"),
                ],
                needle: "こんにちは",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("123"), Some("456"), Some("123"), Some("789")],
                needle: "123",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
            TestCase {
                input: vec![Some("@home"), Some("#rust"), Some("@home"), Some("$money")],
                needle: "@home",
                expected: vec![Some(true), Some(false), Some(true), Some(false)],
            },
        ];

        for case in test_cases {
            let input_array: StringArray = StringArray::from(case.input.clone());

            let etc = EtcStringArray::from_string_array(&input_array, None);

            let result: BooleanArray = etc.compare_equals(case.needle);

            let expected_array: BooleanArray = BooleanArray::from(case.expected.clone());

            assert_eq!(result, expected_array,);
        }
    }
}
