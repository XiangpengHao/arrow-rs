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

use bytes::Bytes;
use criterion::*;
use parquet::encodings::rle::{RleDecoder, RleEncoder};
use rand::prelude::*;

fn generate_rle_heavy_data(size: usize, bit_width: u8) -> Vec<u64> {
    let mut data = Vec::new();
    let mut rng = StdRng::seed_from_u64(42);
    let max_value = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    while data.len() < size {
        let value = rng.random_range(0..=max_value);
        let run_length = rng.random_range(8..=64); // Favor longer runs for RLE
        for _ in 0..run_length.min(size - data.len()) {
            data.push(value);
        }
    }
    data
}

fn generate_bit_packed_data(size: usize, bit_width: u8) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(123);
    let max_value = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    (0..size).map(|_| rng.random_range(0..=max_value)).collect()
}

fn generate_mixed_data(size: usize, bit_width: u8) -> Vec<u64> {
    let mut data = Vec::new();
    let mut rng = StdRng::seed_from_u64(456);
    let max_value = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    while data.len() < size {
        if rng.random_bool(0.5) {
            // Add a run
            let value = rng.random_range(0..=max_value);
            let run_length = rng.random_range(4..=16);
            for _ in 0..run_length.min(size - data.len()) {
                data.push(value);
            }
        } else {
            // Add some random values
            let random_length = rng.random_range(1..=8);
            for _ in 0..random_length.min(size - data.len()) {
                data.push(rng.random_range(0..=max_value));
            }
        }
    }
    data
}

fn encode_data(data: &[u64], bit_width: u8) -> Vec<u8> {
    let mut encoder = RleEncoder::new(bit_width, data.len() * 8);
    for &value in data {
        encoder.put(value);
    }
    encoder.consume()
}

fn bench_rle_decoder_get_batch(c: &mut Criterion) {
    let sizes = [8192, 16384, 32768];
    let bit_widths = [1, 3, 7, 11, 23];

    for &size in &sizes {
        for &bit_width in &bit_widths {
            // Benchmark RLE-heavy data
            let rle_data = generate_rle_heavy_data(size, bit_width);
            let encoded_rle: Bytes = encode_data(&rle_data, bit_width).into();

            let mut group = c.benchmark_group(&format!(
                "rle_batch/rle_heavy/size_{}/width_{}",
                size, bit_width
            ));
            group.throughput(Throughput::Elements(size as u64));
            group.bench_function("decode", |b| {
                b.iter(|| {
                    let mut decoder = RleDecoder::new(bit_width);
                    decoder.set_data(encoded_rle.clone());
                    let mut buffer = vec![0i32; size];
                    decoder.get_batch(&mut buffer).unwrap()
                });
            });
            group.finish();

            // Benchmark bit-packed data
            let bit_packed_data = generate_bit_packed_data(size, bit_width);
            let encoded_bit_packed: Bytes = encode_data(&bit_packed_data, bit_width).into();

            let mut group = c.benchmark_group(&format!(
                "rle_batch/bit_packed/size_{}/width_{}",
                size, bit_width
            ));
            group.throughput(Throughput::Elements(size as u64));
            group.bench_function("decode", |b| {
                b.iter(|| {
                    let mut decoder = RleDecoder::new(bit_width);
                    decoder.set_data(encoded_bit_packed.clone());
                    let mut buffer = vec![0i32; size];
                    decoder.get_batch(&mut buffer).unwrap()
                });
            });
            group.finish();

            // Benchmark mixed data
            let mixed_data = generate_mixed_data(size, bit_width);
            let encoded_mixed: Bytes = encode_data(&mixed_data, bit_width).into();

            let mut group = c.benchmark_group(&format!(
                "rle_batch/mixed/size_{}/width_{}",
                size, bit_width
            ));
            group.throughput(Throughput::Elements(size as u64));
            group.bench_function("decode", |b| {
                b.iter(|| {
                    let mut decoder = RleDecoder::new(bit_width);
                    decoder.set_data(encoded_mixed.clone());
                    let mut buffer = vec![0i32; size];
                    decoder.get_batch(&mut buffer).unwrap()
                });
            });
            group.finish();
        }
    }
}

fn bench_rle_decoder_with_dict(c: &mut Criterion) {
    let size = 16384;
    let bit_widths = [1, 3, 7, 11, 23];

    for &bit_width in &bit_widths {
        let dict_size = 1 << bit_width.min(10); // Reasonable dictionary size
        let dict: Vec<u64> = (0..dict_size).map(|i| i).collect();

        // Generate data with indices that fit within dictionary size
        let effective_bit_width = bit_width.min(10);
        let rle_data = generate_rle_heavy_data(size, effective_bit_width);
        let encoded_rle = encode_data(&rle_data, effective_bit_width);

        let mut group = c.benchmark_group(&format!("rle_with_dict/rle_heavy/width_{}", bit_width));
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function("decode", |b| {
            b.iter(|| {
                let mut decoder = RleDecoder::new(effective_bit_width);
                decoder.set_data(encoded_rle.clone().into());
                let mut buffer = vec![0u64; size];
                decoder
                    .get_batch_with_dict(&dict, &mut buffer, size)
                    .unwrap()
            });
        });
        group.finish();

        let bit_packed_data = generate_bit_packed_data(size, effective_bit_width);
        let encoded_bit_packed = encode_data(&bit_packed_data, effective_bit_width);

        let mut group = c.benchmark_group(&format!("rle_with_dict/bit_packed/width_{}", bit_width));
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function("decode", |b| {
            b.iter(|| {
                let mut decoder = RleDecoder::new(effective_bit_width);
                decoder.set_data(encoded_bit_packed.clone().into());
                let mut buffer = vec![0u64; size];
                decoder
                    .get_batch_with_dict(&dict, &mut buffer, size)
                    .unwrap()
            });
        });
        group.finish();
    }
}

criterion_group!(
    rle_decoder_benches,
    bench_rle_decoder_get_batch,
    bench_rle_decoder_with_dict,
);
criterion_main!(rle_decoder_benches);
