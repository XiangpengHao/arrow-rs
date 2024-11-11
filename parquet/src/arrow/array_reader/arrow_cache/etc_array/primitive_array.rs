use arrow_array::ArrowPrimitiveType;
use arrow_buffer::ArrowNativeType;
use fastlanes::BitPacking;

use super::BitPackedArray;

trait HasUnsignedNativeType: ArrowPrimitiveType {
    type UnSignedType: ArrowNativeType;
}

// pub struct EtcPrimitiveArray<T>
// where
//     T: ArrowPrimitiveType,
//     T::Native: BitPacking + HasUnsignedNativeType,
// {
//     values: BitPackedArray<<T::Native as HasUnsignedNativeType>::UnSignedType>,
//     reference_value: T::Native,
// }
