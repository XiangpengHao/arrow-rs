// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::util::test_util::seedable_rng;
use criterion::*;
use parquet::format::{
    ColumnChunk, ColumnMetaData, CompressionCodec, Encoding, FieldRepetitionType, RowGroup,
    SchemaElement, Type,
};
use parquet::thrift::{TCompactSimdInputProtocol, TCompactSliceInputProtocol, TSerializable};
use rand::Rng;
use thrift::protocol::TCompactOutputProtocol;

const NUM_COLUMNS: usize = 10_000;
const NUM_ROW_GROUPS: usize = 10;

fn encoded_meta() -> Vec<u8> {
    let mut rng = seedable_rng();
    let mut schema = Vec::with_capacity(NUM_COLUMNS + 1);

    schema.push(SchemaElement {
        type_: None,
        type_length: None,
        repetition_type: None,
        name: Default::default(),
        num_children: Some(NUM_COLUMNS as _),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
        logical_type: None,
    });

    for i in 0..NUM_COLUMNS {
        schema.push(SchemaElement {
            type_: Some(Type::FLOAT),
            type_length: None,
            repetition_type: Some(FieldRepetitionType::REQUIRED),
            name: i.to_string().into(),
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        });
    }

    let row_groups = (0..NUM_ROW_GROUPS)
        .map(|i| {
            let columns = (0..NUM_COLUMNS)
                .map(|_| ColumnChunk {
                    file_path: None,
                    file_offset: 0,
                    meta_data: Some(ColumnMetaData {
                        type_: Type::FLOAT,
                        encodings: vec![Encoding::PLAIN, Encoding::RLE_DICTIONARY],
                        path_in_schema: vec![],
                        codec: CompressionCodec::UNCOMPRESSED,
                        num_values: rng.gen(),
                        total_uncompressed_size: rng.gen(),
                        total_compressed_size: rng.gen(),
                        key_value_metadata: None,
                        data_page_offset: rng.gen(),
                        index_page_offset: Some(rng.gen()),
                        dictionary_page_offset: Some(rng.gen()),
                        statistics: None,
                        encoding_stats: None,
                        bloom_filter_offset: None,
                        bloom_filter_length: None,
                    }),
                    offset_index_length: Some(rng.gen()),
                    offset_index_offset: Some(rng.gen()),
                    column_index_length: Some(rng.gen()),
                    column_index_offset: Some(rng.gen()),
                    crypto_metadata: None,
                    encrypted_column_metadata: None,
                })
                .collect();
            RowGroup {
                columns,
                total_byte_size: rng.gen(),
                num_rows: rng.gen(),
                sorting_columns: None,
                file_offset: None,
                total_compressed_size: Some(rng.gen()),
                ordinal: Some(i as _),
            }
        })
        .collect();
    let file = parquet::format::FileMetaData {
        schema,
        row_groups,
        version: 1,
        num_rows: rng.gen(),
        key_value_metadata: None,
        created_by: Some("parquet-rs".into()),
        column_orders: None,
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    };

    let mut buf = Vec::with_capacity(1024);
    {
        let mut out = TCompactOutputProtocol::new(&mut buf);
        file.write_to_out_protocol(&mut out).unwrap();
    }
    buf
}

fn criterion_benchmark(c: &mut Criterion) {
    let buf = black_box(encoded_meta());
    println!("Parquet metadata {}", buf.len());

    c.bench_function("decode full pass", |b| {
        b.iter(|| {
            let mut input = TCompactSliceInputProtocol::new(&buf);
            parquet::format::FileMetaData::read_from_in_protocol(&mut input).unwrap();
        })
    });

    c.bench_function("decode cursor", |b| {
        b.iter(|| {
            let mut input = TCompactSliceInputProtocol::new(&buf);
            parquet::format2::FileMetaData::read_from_in_protocol(&mut input).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
