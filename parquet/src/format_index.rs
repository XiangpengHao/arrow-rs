use thrift::protocol::{field_id, TInputProtocol, TType};

use crate::{
    format2::{
        ColumnCryptoMetaData, ConvertedType, FieldRepetitionType,
        KeyValue, LogicalType, PageEncodingStats, Statistics, Type,
    },
    thrift::TCompactSliceInputProtocol,
};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SchemaElementCursor {
    offset: usize,
}

impl crate::thrift::TSerializable for SchemaElementCursor {
    fn write_to_out_protocol<T: thrift::protocol::TOutputProtocol>(
        &self,
        _o_prot: &mut T,
    ) -> thrift::Result<()> {
        unimplemented!("SchemaElementIndex is not meant to be serialized")
    }

    fn read_from_in_protocol<T: thrift::protocol::TInputProtocol>(
        i_prot: &mut T,
    ) -> thrift::Result<Self> {
        let i_prot = unsafe {
            // very dangerous proof of concept
            std::mem::transmute::<&mut T, &mut TCompactSliceInputProtocol>(i_prot)
        };
        let offset = i_prot.buffer_offset();

        i_prot.read_struct_begin()?;
        loop {
            let field_ident = i_prot.read_field_begin()?;
            if field_ident.field_type == TType::Stop {
                break;
            }
            let field_id = field_id(&field_ident)?;
            match field_id {
                1 => {
                    Type::read_from_in_protocol(i_prot)?;
                }
                2 => {
                    i_prot.read_i32_skip();
                }
                3 => {
                    FieldRepetitionType::read_from_in_protocol(i_prot)?;
                }
                4 => {
                    i_prot.read_string_skip();
                }
                5 => {
                    i_prot.read_i32_skip();
                }
                6 => {
                    ConvertedType::read_from_in_protocol(i_prot)?;
                }
                7 => {
                    i_prot.read_i32_skip();
                }
                8 => {
                    i_prot.read_i32_skip();
                }
                9 => {
                    i_prot.read_i32_skip();
                }
                10 => {
                    LogicalType::read_from_in_protocol(i_prot)?;
                }
                _ => {
                    i_prot.skip(field_ident.field_type)?;
                }
            };
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;

        Ok(SchemaElementCursor { offset })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ColumnMetaDataCursor {
    offset: usize,
}

impl crate::thrift::TSerializable for ColumnMetaDataCursor {
    fn read_from_in_protocol<T: TInputProtocol>(
        i_prot: &mut T,
    ) -> thrift::Result<ColumnMetaDataCursor> {
        let i_prot = unsafe {
            // very dangerous proof of concept
            std::mem::transmute::<&mut T, &mut TCompactSliceInputProtocol>(i_prot)
        };
        let offset = i_prot.buffer_offset();
        i_prot.read_struct_begin()?;
        loop {
            let field_ident = i_prot.read_field_begin()?;
            if field_ident.field_type == TType::Stop {
                break;
            }
            let field_id = field_id(&field_ident)?;
            match field_id {
                1 => {
                    Type::read_from_in_protocol(i_prot)?;
                }
                2 => {
                    let list_ident = i_prot.read_list_begin()?;
                    for _ in 0..list_ident.size {
                        i_prot.read_i32_skip();
                    }
                    i_prot.read_list_end()?;
                }
                3 => {
                    let list_ident = i_prot.read_list_begin()?;
                    for _ in 0..list_ident.size {
                        i_prot.read_string_skip();
                    }
                    i_prot.read_list_end()?;
                }
                4 => {
                    i_prot.read_i32_skip();
                }
                5 => {
                    i_prot.read_i64_skip();
                }
                6 => {
                    i_prot.read_i64_skip();
                }
                7 => {
                    i_prot.read_i64_skip();
                }
                8 => {
                    let list_ident = i_prot.read_list_begin()?;
                    for _ in 0..list_ident.size {
                        KeyValue::read_from_in_protocol(i_prot)?;
                    }
                    i_prot.read_list_end()?;
                }
                9 => {
                    i_prot.read_i64_skip();
                }
                10 => {
                    i_prot.read_i64_skip();
                }
                11 => {
                    i_prot.read_i64_skip();
                }
                12 => {
                    Statistics::read_from_in_protocol(i_prot)?;
                }
                13 => {
                    let list_ident = i_prot.read_list_begin()?;
                    for _ in 0..list_ident.size {
                        PageEncodingStats::read_from_in_protocol(i_prot)?;
                    }
                    i_prot.read_list_end()?;
                }
                14 => {
                    i_prot.read_i64_skip();
                }
                15 => {
                    i_prot.read_i32_skip();
                }
                _ => {
                    i_prot.skip(field_ident.field_type)?;
                }
            };
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(ColumnMetaDataCursor { offset })
    }

    fn write_to_out_protocol<T: thrift::protocol::TOutputProtocol>(
        &self,
        _o_prot: &mut T,
    ) -> thrift::Result<()> {
        unimplemented!("ColumnMetaDataCursor is not meant to be serialized")
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ColumnChunkCursor {
    offset: usize,
}

impl crate::thrift::TSerializable for ColumnChunkCursor {
    fn read_from_in_protocol<T: TInputProtocol>(
        i_prot: &mut T,
    ) -> thrift::Result<ColumnChunkCursor> {
        let i_prot = unsafe {
            // very dangerous proof of concept
            std::mem::transmute::<&mut T, &mut TCompactSliceInputProtocol>(i_prot)
        };
        let offset = i_prot.buffer_offset();
        i_prot.read_struct_begin()?;

        loop {
            let field_ident = i_prot.read_field_begin()?;
            if field_ident.field_type == TType::Stop {
                break;
            }
            let field_id = field_id(&field_ident)?;
            match field_id {
                1 => {
                    i_prot.read_string_skip();
                }
                2 => {
                    i_prot.read_i64_skip();
                }
                3 => {
                    ColumnMetaDataCursor::read_from_in_protocol(i_prot)?;
                }
                4 => {
                    i_prot.read_i64_skip();
                }
                5 => {
                    i_prot.read_i32_skip();
                }
                6 => {
                    i_prot.read_i64_skip();
                }
                7 => {
                    i_prot.read_i32_skip();
                }
                8 => {
                    ColumnCryptoMetaData::read_from_in_protocol(i_prot)?;
                }
                9 => {
                    i_prot.read_bytes_skip();
                }
                _ => {
                    i_prot.skip(field_ident.field_type)?;
                }
            };
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        Ok(ColumnChunkCursor { offset })
    }

    fn write_to_out_protocol<T: thrift::protocol::TOutputProtocol>(
        &self,
        _o_prot: &mut T,
    ) -> thrift::Result<()> {
        unimplemented!()
    }
}
