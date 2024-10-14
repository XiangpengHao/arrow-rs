use crate::arrow::arrow_reader::{RowSelection, RowSelector};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use vortex::arrow::FromArrowArray;
use vortex::IntoCanonical;
use vortex_sampling_compressor::compressors::alp::ALPCompressor;
use vortex_sampling_compressor::compressors::bitpacked::BitPackedCompressor;
use vortex_sampling_compressor::compressors::delta::DeltaCompressor;
use vortex_sampling_compressor::compressors::dict::DictCompressor;
use vortex_sampling_compressor::compressors::fsst::FSSTCompressor;
use vortex_sampling_compressor::compressors::r#for::FoRCompressor;
use vortex_sampling_compressor::compressors::CompressorRef;
use vortex_sampling_compressor::SamplingCompressor;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type as ArrowDate32Type, Date64Type as ArrowDate64Type, Int16Type as ArrowInt16Type,
    Int32Type as ArrowInt32Type, Int64Type as ArrowInt64Type, Int8Type as ArrowInt8Type,
    UInt16Type as ArrowUInt16Type, UInt32Type as ArrowUInt32Type, UInt64Type as ArrowUInt64Type,
    UInt8Type as ArrowUInt8Type,
};
use arrow_array::{
    ArrayRef, DictionaryArray, RecordBatch, RecordBatchWriter, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

static ARROW_ARRAY_CACHE: LazyLock<ArrowArrayCache> = LazyLock::new(|| ArrowArrayCache::new());

/// Row offset -> (Arrow Array, hit count)
type RowMapping = HashMap<usize, CachedEntry>;

/// Column offset -> RowMapping
type ColumnMapping = HashMap<usize, RowMapping>;

#[derive(Debug, Clone)]
enum ArrowCacheMode {
    InMemory,
    OnDisk,
    NoCache,
    Vortex(SamplingCompressor<'static>),
}

struct CachedEntry {
    value: CachedValue,
    row_count: u32,
    hit_count: AtomicU32,
}

impl CachedEntry {
    fn in_memory(array: ArrayRef) -> Self {
        let len = array.len();
        let val = CachedValue::InMemory(array);
        CachedEntry {
            value: val,
            row_count: len as u32,
            hit_count: AtomicU32::new(0),
        }
    }

    fn on_disk(path: String, row_count: usize) -> Self {
        let val = CachedValue::OnDisk(path);
        CachedEntry {
            value: val,
            row_count: row_count as u32,
            hit_count: AtomicU32::new(0),
        }
    }

    fn vortex(array: Arc<vortex::Array>) -> Self {
        let len = array.len();
        let val = CachedValue::Vortex(array);
        CachedEntry {
            value: val,
            row_count: len as u32,
            hit_count: AtomicU32::new(0),
        }
    }
}

enum CachedValue {
    InMemory(ArrayRef),
    Vortex(Arc<vortex::Array>),
    OnDisk(String),
}

impl CachedValue {
    fn memory_usage(&self) -> usize {
        match self {
            Self::InMemory(array) => array.get_array_memory_size(),
            Self::OnDisk(_) => 0,
            Self::Vortex(array) => array.nbytes(),
        }
    }
}

/// ArrayIdentifier is used to identify an array in the cache.
pub struct ArrayIdentifier {
    row_group_id: usize,
    column_id: usize,
    row_id: usize, // followed by the batch size
}

impl ArrayIdentifier {
    /// Create a new ArrayIdentifier.
    pub fn new(row_group_id: usize, column_id: usize, row_id: usize) -> Self {
        Self {
            row_group_id,
            column_id,
            row_id,
        }
    }
}

/// ArrowCacheStatistics is used to collect statistics about the arrow array cache.
#[derive(Debug, serde::Serialize)]
pub struct ArrowCacheStatistics {
    /// Row group ids
    pub row_group_ids: Vec<u64>,
    /// Column ids
    pub column_ids: Vec<u64>,
    /// Row start ids
    pub row_start_ids: Vec<u64>,
    /// Row counts
    pub row_counts: Vec<u64>,
    /// Memory sizes
    pub memory_sizes: Vec<u64>,
    /// Cache types
    pub cache_types: Vec<CacheType>,
    /// Hit counts
    pub hit_counts: Vec<u64>,
}

impl ArrowCacheStatistics {
    /// Create a new ArrowCacheStatistics.
    pub fn new() -> Self {
        ArrowCacheStatistics {
            row_group_ids: Vec::new(),
            column_ids: Vec::new(),
            row_start_ids: Vec::new(),
            row_counts: Vec::new(),
            memory_sizes: Vec::new(),
            cache_types: Vec::new(),
            hit_counts: Vec::new(),
        }
    }

    /// Add an entry to the statistics.
    pub fn add_entry(
        &mut self,
        row_group_id: u64,
        column_id: u64,
        row_start_id: u64,
        row_count: u64,
        memory_size: u64,
        cache_type: CacheType,
        hit_count: u64,
    ) {
        self.row_group_ids.push(row_group_id);
        self.column_ids.push(column_id);
        self.row_start_ids.push(row_start_id);
        self.row_counts.push(row_count);
        self.memory_sizes.push(memory_size);
        self.cache_types.push(cache_type);
        self.hit_counts.push(hit_count);
    }

    /// Get the total memory usage of the cache.
    pub fn memory_usage(&self) -> usize {
        self.memory_sizes.iter().sum::<u64>() as usize
    }

    /// Convert the statistics to a record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        let row_group_ids = Arc::new(UInt64Array::from_iter(self.row_group_ids));
        let column_ids = Arc::new(UInt64Array::from(self.column_ids));
        let row_start_ids = Arc::new(UInt64Array::from(self.row_start_ids));
        let row_counts = Arc::new(UInt64Array::from(self.row_counts));
        let memory_sizes = Arc::new(UInt64Array::from(self.memory_sizes));

        let cache_type_values: Vec<&str> = self
            .cache_types
            .iter()
            .map(|ct| match ct {
                CacheType::InMemory => "InMemory",
                CacheType::OnDisk => "OnDisk",
                CacheType::Vortex => "Vortex",
            })
            .collect();
        let cache_types = Arc::new(DictionaryArray::new(
            UInt32Array::from((0..self.cache_types.len() as u32).collect::<Vec<u32>>()),
            Arc::new(StringArray::from(cache_type_values)),
        ));

        let hit_counts = Arc::new(UInt64Array::from(self.hit_counts.clone()));

        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("row_group_id", DataType::UInt64, false),
                Field::new("column_id", DataType::UInt64, false),
                Field::new("row_start_id", DataType::UInt64, false),
                Field::new("row_count", DataType::UInt64, false),
                Field::new("memory_size", DataType::UInt64, false),
                Field::new(
                    "cache_type",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    false,
                ),
                Field::new("hit_count", DataType::UInt64, false),
            ])),
            vec![
                row_group_ids,
                column_ids,
                row_start_ids,
                row_counts,
                memory_sizes,
                cache_types,
                hit_counts,
            ],
        )
        .unwrap()
    }
}

/// CacheType is used to identify the type of cache.
#[derive(Debug, serde::Serialize)]
pub enum CacheType {
    /// InMemory cache
    InMemory,
    /// OnDisk cache
    OnDisk,
    /// Vortex cache
    Vortex,
}

/// ArrowArrayCache is used to cache arrow arrays in memory, on disk, or in a vortex.
pub struct ArrowArrayCache {
    /// Vec of RwLocks, where index is the row group index and value is the ColumnMapping
    value: Vec<RwLock<ColumnMapping>>,
    cache_mode: ArrowCacheMode,
}

impl ArrowArrayCache {
    /// Create a new ArrowArrayCache.
    fn new() -> Self {
        const MAX_ROW_GROUPS: usize = 512;
        ArrowArrayCache {
            value: (0..MAX_ROW_GROUPS)
                .map(|_| RwLock::new(ColumnMapping::new()))
                .collect(),
            cache_mode: std::env::var("ARROW_CACHE_MODE").map_or(ArrowCacheMode::NoCache, |v| {
                match v.to_lowercase().as_str() {
                    "disk" => ArrowCacheMode::OnDisk,
                    "inmemory" => ArrowCacheMode::InMemory,
                    "nocache" => ArrowCacheMode::NoCache,
                    "vortex" => ArrowCacheMode::Vortex(SamplingCompressor::new_with_options(
                        HashSet::from([
                            &ALPCompressor as CompressorRef,
                            &BitPackedCompressor,
                            &DeltaCompressor,
                            &DictCompressor,
                            &FoRCompressor,
                            &FSSTCompressor,
                        ]),
                        Default::default(),
                    )),
                    _ => panic!(
                        "Invalid cache mode: {}, must be one of [disk, inmemory, nocache, vortex]",
                        v
                    ),
                }
            }),
        }
    }

    /// Get the static ArrowArrayCache.
    pub fn get() -> &'static ArrowArrayCache {
        &ARROW_ARRAY_CACHE
    }

    /// Check if the cache is enabled.
    pub fn cache_enabled(&self) -> bool {
        !matches!(self.cache_mode, ArrowCacheMode::NoCache)
    }

    /// Reset the cache.
    pub fn reset(&self) {
        for row_group in self.value.iter() {
            let mut row_group = row_group.write().unwrap();
            row_group.clear();
        }
    }

    /// Returns a list of ranges that are cached.
    /// The ranges are sorted by the starting row id.
    pub fn get_cached_ranges_of_column(
        &self,
        row_group_id: usize,
        column_id: usize,
    ) -> Option<HashSet<Range<usize>>> {
        let v = &self.value[row_group_id].read().unwrap();
        let rows = v.get(&column_id)?;

        let mut result_ranges = HashSet::new();
        for (row_id, cached_entry) in rows.iter() {
            let start = *row_id;
            let end = row_id + cached_entry.row_count as usize;

            result_ranges.insert(start..end);
        }
        Some(result_ranges)
    }

    /// Get coalesced record batches from a row selection.
    /// This function is semantically the same as `get_record_batch`,
    /// but it tries to coalesce the record batches so that upper query engine can
    /// read the record batches in a more efficient way.
    ///
    /// Note that if the selection is not sparse, this function can be slower.
    pub fn get_coalesced_record_batches(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        column_idx: &[usize],
    ) -> Vec<RecordBatch> {
        let mut record_batches = vec![];
        let mut row_id = 0;

        let mut row_idx_of_current_batch: (usize, Vec<u32>) = (0, vec![]); // (batch_id, row_idx)
        for row_selector in selection.iter() {
            if row_selector.skip {
                row_id += row_selector.row_count;
                continue;
            }
            for j in 0..row_selector.row_count {
                let new_row_idx = row_id + j;
                if (new_row_idx / 8192 * 8192) != row_idx_of_current_batch.0 {
                    if !row_idx_of_current_batch.1.is_empty() {
                        let batch_id = row_idx_of_current_batch.0;
                        let mut columns = Vec::with_capacity(column_idx.len());
                        let indices = UInt32Array::from(row_idx_of_current_batch.1.clone());
                        for &column_id in column_idx {
                            let id = ArrayIdentifier::new(row_group_id, column_id, batch_id);
                            let mut array = self.get_arrow_array(&id).unwrap();
                            array = arrow_select::take::take(&array, &indices, None).unwrap();
                            columns.push(array);
                        }
                        let record_batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
                        record_batches.push(record_batch);
                        row_idx_of_current_batch.1.clear();
                    }
                    row_idx_of_current_batch.0 = new_row_idx / 8192 * 8192;
                }
                row_idx_of_current_batch
                    .1
                    .push((new_row_idx - row_idx_of_current_batch.0) as u32);
            }
            row_id += row_selector.row_count;
        }

        if !row_idx_of_current_batch.1.is_empty() {
            let mut columns = Vec::with_capacity(column_idx.len());
            let indices = UInt32Array::from(row_idx_of_current_batch.1.clone());
            for &column_id in column_idx {
                let id = ArrayIdentifier::new(row_group_id, column_id, row_idx_of_current_batch.0);
                let mut array = self.get_arrow_array(&id).unwrap();
                array = arrow_select::take::take(&array, &indices, None).unwrap();
                columns.push(array);
            }
            let record_batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
            record_batches.push(record_batch);
        }

        let mut coalesced_batches = vec![];
        let mut accumulated = (0, 0); // (num_rows, batch_idx)
        for i in 0..record_batches.len() {
            let batch = &record_batches[i];
            let len = batch.num_rows();
            if accumulated.0 + len <= 8192 {
                accumulated.0 += len;
            } else {
                let prev_batches = &record_batches[accumulated.1..i];
                let new_batch =
                    arrow_select::concat::concat_batches(&schema, prev_batches).unwrap();
                coalesced_batches.push(new_batch);
                accumulated = (len, i);
            }
        }
        if accumulated.0 > 0 {
            let prev_batches = &record_batches[accumulated.1..];
            let new_batch = arrow_select::concat::concat_batches(&schema, prev_batches).unwrap();
            coalesced_batches.push(new_batch);
        }
        coalesced_batches
    }

    /// Get a record batch from a row selection.
    pub fn get_record_batches(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> Vec<RecordBatch> {
        let selection_count = selection.selectors().len();
        let total_row_count = selection.iter().map(|s| s.row_count).sum::<usize>();
        let is_sparse = (total_row_count / 16) < selection_count;
        // estimate sparsity, if sparse,
        // we should try to coalesce the selection (which has overhead of its own)
        // otherwise, we just emit the sliced record batches
        let record_batches = if is_sparse {
            ArrowArrayCache::get().get_coalesced_record_batches(
                row_group_id,
                &selection,
                &schema,
                &parquet_column_ids,
            )
        } else {
            ArrowArrayCache::get().get_record_batch_by_slice(
                row_group_id,
                &selection,
                &schema,
                &parquet_column_ids,
            )
        };
        record_batches
    }

    /// Get a record batch iterator from a row selection.
    pub fn get_record_batch_by_slice_iter(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> RecordBatchBySliceIterator {
        RecordBatchBySliceIterator {
            row_group_id,
            selection: selection.clone().into(),
            schema: schema.clone(),
            parquet_column_ids: parquet_column_ids.to_vec(),
            row_id: 0,
            current_selected: 0,
        }
    }

    /// Get a record batch from a row selection.
    pub fn get_record_batch_by_slice(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> Vec<RecordBatch> {
        let mut record_batches = Vec::new();

        let mut row_id = 0;

        for row_selector in selection.iter() {
            if row_selector.skip {
                row_id += row_selector.row_count;
                continue;
            }

            let mut current_selected = 0;
            while current_selected < row_selector.row_count {
                let current_row_id = row_id + current_selected;
                let batch_row_id = current_row_id / 8192 * 8192; // TODO: batch size may not be 8192
                let batch_row_count = 8192 - (current_row_id % 8192);
                let offset = current_row_id.saturating_sub(batch_row_id);
                let want_to_select =
                    std::cmp::min(batch_row_count, row_selector.row_count - current_selected);

                let mut columns = Vec::new();
                for &column_id in parquet_column_ids {
                    let id = ArrayIdentifier::new(row_group_id, column_id, batch_row_id);
                    let mut array = self.get_arrow_array(&id).unwrap();
                    if offset > 0 || want_to_select < array.len() {
                        array = array.slice(offset, want_to_select);
                    }
                    columns.push(array);
                }

                assert!(columns.len() == parquet_column_ids.len());
                let record_batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

                assert_eq!(record_batch.num_rows(), want_to_select);

                record_batches.push(record_batch);

                current_selected += want_to_select;
            }

            assert_eq!(current_selected, row_selector.row_count);

            row_id += row_selector.row_count;
        }
        record_batches
    }

    /// Returns the cached ranges for the given row group and column indices.
    ///
    /// It returns the **intersection** of the cached ranges for the given column indices.
    /// If any column indices' cached ranges are not found, the function returns `None`.
    pub fn get_cached_ranges_of_columns(
        &self,
        row_group_id: usize,
        column_ids: &[usize],
    ) -> Option<HashSet<Range<usize>>> {
        let mut intersected_ranges: Option<HashSet<Range<usize>>> = None;

        for column_idx in column_ids.iter() {
            if let Some(cached_ranges) = self.get_cached_ranges_of_column(row_group_id, *column_idx)
            {
                intersected_ranges = match intersected_ranges {
                    None => Some(cached_ranges),
                    Some(existing_ranges) => {
                        Some(intersect_ranges(existing_ranges, &cached_ranges))
                    }
                };
            } else {
                return None;
            }
        }
        intersected_ranges
    }

    /// Get an arrow array from the cache.
    pub fn get_arrow_array(&self, id: &ArrayIdentifier) -> Option<ArrayRef> {
        if matches!(self.cache_mode, ArrowCacheMode::NoCache) {
            return None;
        }

        let cache = &self.value[id.row_group_id].read().unwrap();

        let column_cache = cache.get(&id.column_id)?;
        let cached_entry = column_cache.get(&id.row_id)?;
        cached_entry.hit_count.fetch_add(1, Ordering::Relaxed);
        match &cached_entry.value {
            CachedValue::InMemory(array) => Some(array.clone()),
            CachedValue::OnDisk(path) => {
                let file = std::fs::File::open(path).ok()?;
                let reader = std::io::BufReader::new(file);
                let mut arrow_reader = FileReader::try_new(reader, None).ok()?;
                let batch = arrow_reader.next().unwrap().unwrap();
                Some(batch.column(0).clone())
            }
            CachedValue::Vortex(array) => {
                let array: vortex::Array = (**array).clone();
                let array = array.into_canonical().unwrap();
                let canonical_array = array.into_arrow().unwrap();
                Some(canonical_array)
            }
        }
    }

    fn id_exists(&self, id: &ArrayIdentifier) -> bool {
        let cache = &self.value[id.row_group_id].read().unwrap();
        cache.contains_key(&id.column_id)
            && cache.get(&id.column_id).unwrap().contains_key(&id.row_id)
    }

    /// Insert an arrow array into the cache.
    pub(crate) fn insert_arrow_array(&self, id: &ArrayIdentifier, array: ArrayRef) {
        if matches!(self.cache_mode, ArrowCacheMode::NoCache) {
            return;
        }

        if self.id_exists(id) {
            return;
        }

        let mut cache = self.value[id.row_group_id].write().unwrap();

        let column_cache = cache.entry(id.column_id).or_insert_with(HashMap::new);

        match &self.cache_mode {
            ArrowCacheMode::InMemory => {
                let old = column_cache.insert(id.row_id, CachedEntry::in_memory(array));
                assert!(old.is_none());
            }
            ArrowCacheMode::OnDisk => {
                let path = format!(
                    "target/arrow-cache/arrow_array_{}_{}_{}.arrow",
                    id.row_group_id, id.column_id, id.row_id
                );
                std::fs::create_dir_all("target/arrow-cache").unwrap();
                let file = std::fs::File::create(path.clone()).unwrap();
                let mut writer = std::io::BufWriter::new(file);
                let schema = Schema::new(vec![Field::new(
                    "_",
                    array.data_type().clone(),
                    array.is_nullable(),
                )]);
                let array_len = array.len();
                let record_batch =
                    RecordBatch::try_new(Arc::new(schema.clone()), vec![array]).unwrap();
                let mut arrow_writer = FileWriter::try_new(&mut writer, &schema).unwrap();
                arrow_writer.write(&record_batch).unwrap();
                arrow_writer.close().unwrap();
                column_cache.insert(id.row_id, CachedEntry::on_disk(path, array_len));
            }
            ArrowCacheMode::Vortex(compressor) => {
                let data_type = array.data_type();

                if array.len() < 8192 {
                    // our batch is too small.
                    column_cache.insert(id.row_id, CachedEntry::in_memory(array));
                    return;
                }

                let vortex_array = match data_type {
                    DataType::Date32 => {
                        let primitive_array = array.as_primitive::<ArrowDate32Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::Date64 => {
                        let primitive_array = array.as_primitive::<ArrowDate64Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::UInt64 => {
                        let primitive_array = array.as_primitive::<ArrowUInt64Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::UInt32 => {
                        let primitive_array = array.as_primitive::<ArrowUInt32Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::UInt16 => {
                        let primitive_array = array.as_primitive::<ArrowUInt16Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::UInt8 => {
                        let primitive_array = array.as_primitive::<ArrowUInt8Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::Int64 => {
                        let primitive_array = array.as_primitive::<ArrowInt64Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::Int32 => {
                        let primitive_array = array.as_primitive::<ArrowInt32Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::Int16 => {
                        let primitive_array = array.as_primitive::<ArrowInt16Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::Int8 => {
                        let primitive_array = array.as_primitive::<ArrowInt8Type>();
                        vortex::Array::from_arrow(primitive_array, array.logical_nulls().is_some())
                    }
                    DataType::Utf8 => {
                        let string_array = array.as_string::<i32>();
                        vortex::Array::from_arrow(string_array, array.logical_nulls().is_some())
                    }
                    _ => {
                        unimplemented!("data type {:?} not implemented", data_type);
                    }
                };

                match compressor.compress(&vortex_array, None) {
                    Ok(compressed) => {
                        column_cache.insert(
                            id.row_id,
                            CachedEntry::vortex(Arc::new(compressed.into_array())),
                        );
                    }
                    Err(_e) => {
                        column_cache.insert(id.row_id, CachedEntry::in_memory(array));
                    }
                }
            }

            ArrowCacheMode::NoCache => {
                unreachable!()
            }
        }
    }

    /// Collect statistics about the cache.
    pub fn stats(&self) -> ArrowCacheStatistics {
        let mut stats = ArrowCacheStatistics::new();

        for (row_group_id, row_group_lock) in self.value.iter().enumerate() {
            let row_group = row_group_lock.read().unwrap();

            for (column_id, row_mapping) in row_group.iter() {
                for (row_start_id, cached_entry) in row_mapping {
                    let cache_type = match cached_entry.value {
                        CachedValue::InMemory(_) => CacheType::InMemory,
                        CachedValue::OnDisk(_) => CacheType::OnDisk,
                        CachedValue::Vortex(_) => CacheType::Vortex,
                    };

                    let memory_size = cached_entry.value.memory_usage();
                    let row_count = match &cached_entry.value {
                        CachedValue::InMemory(array) => array.len(),
                        CachedValue::OnDisk(_) => 0, // We don't know the row count for on-disk entries
                        CachedValue::Vortex(array) => array.len(),
                    };

                    stats.add_entry(
                        row_group_id as u64,
                        *column_id as u64,
                        *row_start_id as u64,
                        row_count as u64,
                        memory_size as u64,
                        cache_type,
                        cached_entry.hit_count.load(Ordering::Relaxed) as u64,
                    );
                }
            }
        }

        stats
    }
}

/// Iterator over `RecordBatch` for `get_record_batch_by_slice`.
pub struct RecordBatchBySliceIterator {
    row_group_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    parquet_column_ids: Vec<usize>,
    row_id: usize,
    current_selected: usize,
}

impl Iterator for RecordBatchBySliceIterator {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(row_selector) = self.selection.pop_front() {
            if row_selector.skip {
                self.row_id += row_selector.row_count;
                continue;
            }

            assert!(self.current_selected < row_selector.row_count);
            let current_row_id = self.row_id + self.current_selected;
            let batch_row_id = (current_row_id / 8192) * 8192;
            let batch_row_count = 8192 - (current_row_id % 8192);
            let offset = current_row_id.saturating_sub(batch_row_id);
            let want_to_select = std::cmp::min(
                batch_row_count,
                row_selector.row_count - self.current_selected,
            );

            let mut columns = Vec::with_capacity(self.parquet_column_ids.len());
            for &column_id in &self.parquet_column_ids {
                let id = ArrayIdentifier::new(self.row_group_id, column_id, batch_row_id);
                let mut array = ArrowArrayCache::get().get_arrow_array(&id)?;
                if offset > 0 || want_to_select < array.len() {
                    array = array.slice(offset, want_to_select);
                }
                columns.push(array);
            }

            assert_eq!(columns.len(), self.parquet_column_ids.len());
            let record_batch = RecordBatch::try_new(self.schema.clone(), columns).unwrap();

            assert_eq!(record_batch.num_rows(), want_to_select);

            self.current_selected += want_to_select;
            self.row_id += want_to_select;
            if self.current_selected < row_selector.row_count {
                self.selection.push_front(row_selector);
            } else {
                self.current_selected = 0;
            }
            return Some(record_batch);
        }
        None
    }
}

/// Returns the ranges that are present in both `base` and `input`.
/// Ranges in both sets are assumed to be non-overlapping.
///
/// The returned ranges are exactly those that appear in both input sets,
/// preserving their original bounds for use in cache retrieval.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use std::ops::Range;
///
/// let base: HashSet<Range<usize>> = vec![0..10, 20..30, 40..45].into_iter().collect();
/// let input: HashSet<Range<usize>> = vec![0..10, 25..35, 40..45].into_iter().collect();
/// let result = intersect_ranges(base, input);
/// assert_eq!(result, vec![0..10, 40..45].into_iter().collect::<HashSet<_>>());
/// ```
fn intersect_ranges(
    mut base: HashSet<Range<usize>>,
    input: &HashSet<Range<usize>>,
) -> HashSet<Range<usize>> {
    base.retain(|range| input.contains(range));
    base
}
