mod bit_pack_array;
mod fsst_array;
mod primitive_array;
mod string_array;

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, BooleanArray};

use bit_pack_array::BitPackedArray;
use fastlanes::BitPacking;
use fsst_array::FsstArray;

use primitive_array::HasUnsignedType;
pub use primitive_array::{EtcPrimitiveArray, EtcPrimitiveMetadata};
pub use string_array::{EtcStringArray, EtcStringMetadata};

pub trait AsEtcArray {
    fn as_string_array_opt(&self) -> Option<&EtcStringArray>;

    fn as_string(&self) -> &EtcStringArray {
        self.as_string_array_opt().expect("etc string array")
    }

    fn as_primitive_array_opt<T: ArrowPrimitiveType + HasUnsignedType>(
        &self,
    ) -> Option<&EtcPrimitiveArray<T>>
    where
        <<T as HasUnsignedType>::UnSignedType as ArrowPrimitiveType>::Native: BitPacking;

    fn as_primitive<T: ArrowPrimitiveType + HasUnsignedType>(&self) -> &EtcPrimitiveArray<T>
    where
        <<T as HasUnsignedType>::UnSignedType as ArrowPrimitiveType>::Native: BitPacking,
    {
        self.as_primitive_array_opt().expect("etc primitive array")
    }
}

impl AsEtcArray for dyn Array + '_ {
    fn as_string_array_opt(&self) -> Option<&EtcStringArray> {
        self.as_any().downcast_ref()
    }

    fn as_primitive_array_opt<T: ArrowPrimitiveType + HasUnsignedType>(
        &self,
    ) -> Option<&EtcPrimitiveArray<T>>
    where
        <<T as HasUnsignedType>::UnSignedType as ArrowPrimitiveType>::Native: BitPacking,
    {
        self.as_any().downcast_ref()
    }
}

pub trait EtcArray: std::fmt::Debug + Send + Sync {
    fn get_array_memory_size(&self) -> usize;

    fn len(&self) -> usize;

    fn to_arrow_array(&self) -> ArrayRef;

    fn filter(&self, selection: &BooleanArray) -> EtcArrayRef;
}

pub type EtcArrayRef = Arc<dyn EtcArray>;

pub(crate) fn get_bit_width(max_value: u64) -> u8 {
    if max_value <= 1 {
        0
    } else {
        64 - max_value.leading_zeros() as u8
    }
}
