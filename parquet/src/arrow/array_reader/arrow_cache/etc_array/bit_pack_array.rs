use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_data::ArrayDataBuilder;
use fastlanes::BitPacking;

pub(crate) struct BitPackedArray<T: ArrowPrimitiveType>
where
    T::Native: BitPacking,
{
    pub(crate) values: PrimitiveArray<T>,
    pub(crate) bit_width: u8,
    pub(crate) original_len: usize,
}

impl<T: ArrowPrimitiveType> BitPackedArray<T>
where
    T::Native: BitPacking,
{
    pub fn from_parts(values: PrimitiveArray<T>, bit_width: u8, original_len: usize) -> Self {
        Self {
            values,
            bit_width,
            original_len,
        }
    }

    pub fn len(&self) -> usize {
        self.original_len
    }

    pub fn from_primitive(array: PrimitiveArray<T>, bit_width: u8) -> Self {
        let original_len = array.len();
        let (_data_type, values, nulls) = array.into_parts();

        let bit_width = bit_width as usize;
        let num_chunks = (original_len + 1023) / 1024;
        let num_full_chunks = original_len / 1024;
        let packed_len =
            (1024 * bit_width + size_of::<T::Native>() * 8 - 1) / (size_of::<T::Native>() * 8);

        let mut output = Vec::<T::Native>::with_capacity(num_chunks * packed_len);

        (0..num_full_chunks).for_each(|i| {
            let start_elem = i * 1024;

            output.reserve(packed_len);
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_len);
                BitPacking::unchecked_pack(
                    bit_width,
                    &values[start_elem..][..1024],
                    &mut output[output_len..][..packed_len],
                );
            }
        });

        if num_chunks != num_full_chunks {
            let last_chunk_size = values.len() % 1024;
            let mut last_chunk = vec![T::Native::default(); 1024];
            last_chunk[..last_chunk_size]
                .copy_from_slice(&values[values.len() - last_chunk_size..]);

            output.reserve(packed_len);
            let output_len = output.len();
            unsafe {
                output.set_len(output_len + packed_len);
                BitPacking::unchecked_pack(
                    bit_width,
                    &last_chunk,
                    &mut output[output_len..][..packed_len],
                );
            }
        }

        let scalar_buffer = Buffer::from(output);
        let array_builder = unsafe {
            ArrayDataBuilder::new(T::DATA_TYPE)
                .len(num_chunks * packed_len)
                .add_buffer(scalar_buffer)
                .nulls(nulls)
                .build_unchecked()
        };

        let values = PrimitiveArray::<T>::from(array_builder);

        Self {
            values,
            bit_width: bit_width as u8,
            original_len,
        }
    }

    pub(crate) fn to_primitive(&self) -> PrimitiveArray<T> {
        let bit_width = self.bit_width as usize;
        let packed = self.values.values().as_ref();
        let length = self.original_len;
        let offset = 0;

        let num_chunks = (offset + length + 1023) / 1024;
        let elements_per_chunk =
            (1024 * bit_width + size_of::<T::Native>() * 8 - 1) / (size_of::<T::Native>() * 8);

        let mut output = Vec::<T::Native>::with_capacity(num_chunks * 1024 - offset);

        let first_full_chunk = if offset != 0 {
            let chunk: &[T::Native] = &packed[0..elements_per_chunk];
            let mut decoded = vec![T::Native::default(); 1024];
            unsafe { BitPacking::unchecked_unpack(bit_width, chunk, &mut decoded) };
            output.extend_from_slice(&decoded[offset..]);
            1
        } else {
            0
        };

        (first_full_chunk..num_chunks).for_each(|i| {
            let chunk: &[T::Native] = &packed[i * elements_per_chunk..][0..elements_per_chunk];
            unsafe {
                let output_len = output.len();
                output.set_len(output_len + 1024);
                BitPacking::unchecked_unpack(bit_width, chunk, &mut output[output_len..][..1024]);
            }
        });

        output.truncate(length);
        if output.len() < 1024 {
            output.shrink_to_fit();
        }

        let nulls = self.values.nulls().cloned();
        PrimitiveArray::<T>::new(ScalarBuffer::from(output), nulls)
    }

    pub(crate) fn get_array_memory_size(&self) -> usize {
        self.values.get_array_memory_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::UInt32Type;

    #[test]
    fn test_bit_pack_roundtrip() {
        // Test with a full chunk (1024 elements)
        let values: Vec<u32> = (0..1024).collect();

        let array = PrimitiveArray::<UInt32Type>::from(values);
        let before_size = array.get_array_memory_size();
        let bit_packed = BitPackedArray::from_primitive(array, 10);
        let after_size = bit_packed.get_array_memory_size();
        println!("before: {}, after: {}", before_size, after_size);
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 1024);
        for i in 0..1024 {
            assert_eq!(unpacked.value(i), i as u32);
        }
    }

    #[test]
    fn test_bit_pack_partial_chunk() {
        // Test with a partial chunk (500 elements)
        let values: Vec<u32> = (0..500).collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);
        let bit_packed = BitPackedArray::from_primitive(array, 10);
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 500);
        for i in 0..500 {
            assert_eq!(unpacked.value(i), i as u32);
        }
    }

    #[test]
    fn test_bit_pack_multiple_chunks() {
        // Test with multiple chunks (2048 elements = 2 full chunks)
        let values: Vec<u32> = (0..2048).collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);
        let bit_packed = BitPackedArray::from_primitive(array, 11);
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 2048);
        for i in 0..2048 {
            assert_eq!(unpacked.value(i), i as u32);
        }
    }

    #[test]
    fn test_bit_pack_with_nulls() {
        let values: Vec<Option<u32>> = (0..1000)
            .map(|i| if i % 2 == 0 { Some(i as u32) } else { None })
            .collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);
        let bit_packed = BitPackedArray::from_primitive(array, 10);
        let unpacked = bit_packed.to_primitive();

        assert_eq!(unpacked.len(), 1000);
        for i in 0..1000 {
            if i % 2 == 0 {
                assert_eq!(unpacked.value(i), i as u32);
            } else {
                assert!(unpacked.is_null(i));
            }
        }
    }

    #[test]
    fn test_different_bit_widths() {
        // Test with different bit widths
        let values: Vec<u32> = (0..100).map(|i| i * 2).collect();
        let array = PrimitiveArray::<UInt32Type>::from(values);

        for bit_width in [8, 16, 24, 32] {
            let bit_packed = BitPackedArray::from_primitive(array.clone(), bit_width);
            let unpacked = bit_packed.to_primitive();

            assert_eq!(unpacked.len(), 100);
            for i in 0..100 {
                assert_eq!(unpacked.value(i), i as u32 * 2);
            }
        }
    }
}
