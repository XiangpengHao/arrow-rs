mod bit_pack_array;
mod fsst_array;
mod primitive_array;
mod string_array;

use std::{any::Any, sync::Arc};

use arrow_array::{ArrayRef, ArrowPrimitiveType, BooleanArray};

use arrow_schema::Schema;
use bit_pack_array::BitPackedArray;
use fastlanes::BitPacking;
use fsst_array::FsstArray;

use primitive_array::HasUnsignedType;
pub use primitive_array::{EtcPrimitiveArray, EtcPrimitiveMetadata};
pub use string_array::{EtcStringArray, EtcStringMetadata};

/// A trait to access the underlying ETC array.
pub trait AsEtcArray {
    /// Get the underlying string array.
    fn as_string_array_opt(&self) -> Option<&EtcStringArray>;

    /// Get the underlying string array.
    fn as_string(&self) -> &EtcStringArray {
        self.as_string_array_opt().expect("etc string array")
    }

    /// Get the underlying primitive array.
    fn as_primitive_array_opt<T: ArrowPrimitiveType + HasUnsignedType>(
        &self,
    ) -> Option<&EtcPrimitiveArray<T>>
    where
        <<T as HasUnsignedType>::UnSignedType as ArrowPrimitiveType>::Native: BitPacking;

    /// Get the underlying primitive array.
    fn as_primitive<T: ArrowPrimitiveType + HasUnsignedType>(&self) -> &EtcPrimitiveArray<T>
    where
        <<T as HasUnsignedType>::UnSignedType as ArrowPrimitiveType>::Native: BitPacking,
    {
        self.as_primitive_array_opt().expect("etc primitive array")
    }
}

impl AsEtcArray for dyn EtcArray + '_ {
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

/// An ETC array.
pub trait EtcArray: std::fmt::Debug + Send + Sync {
    /// Get the underlying any type.
    fn as_any(&self) -> &dyn Any;

    /// Get the memory size of the ETC array.
    fn get_array_memory_size(&self) -> usize;

    /// Get the length of the ETC array.
    fn len(&self) -> usize;

    /// Convert the ETC array to an Arrow array.
    fn to_arrow_array(&self) -> (ArrayRef, Schema);

    /// Filter the ETC array with a boolean array.
    fn filter(&self, selection: &BooleanArray) -> EtcArrayRef;
}

/// A reference to an ETC array.
pub type EtcArrayRef = Arc<dyn EtcArray>;

pub(crate) fn get_bit_width(max_value: u64) -> u8 {
    if max_value <= 1 {
        0
    } else {
        64 - max_value.leading_zeros() as u8
    }
}
