use crate::arrow::arrow_reader::{BooleanSelection, RowSelection, RowSelector};
use ahash::AHashMap;
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use arrow_select::concat::concat_batches;
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
    ArrayRef, DictionaryArray, RecordBatch, RecordBatchWriter, StringArray, StructArray,
    UInt32Array, UInt64Array,
};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::collections::{HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

static ARROW_ARRAY_CACHE: LazyLock<ArrowArrayCache> =
    LazyLock::new(|| ArrowArrayCache::initialize_from_env());

/// Row offset -> (Arrow Array, hit count)
type RowMapping = AHashMap<usize, CachedEntry>;

/// Column offset -> RowMapping
type ColumnMapping = AHashMap<usize, RowMapping>;

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
    batch_size: usize,
}

impl ArrowArrayCache {
    fn initialize_from_env() -> Self {
        let cache_mode = std::env::var("ARROW_CACHE_MODE").map_or(ArrowCacheMode::NoCache, |v| {
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
        });
        ArrowArrayCache::new(cache_mode, 8192)
    }

    /// Create a new ArrowArrayCache.
    fn new(cache_mode: ArrowCacheMode, batch_size: usize) -> Self {
        assert!(batch_size.is_power_of_two());
        const MAX_ROW_GROUPS: usize = 512;
        ArrowArrayCache {
            value: (0..MAX_ROW_GROUPS)
                .map(|_| RwLock::new(ColumnMapping::new()))
                .collect(),
            cache_mode,
            batch_size,
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

    /// Get a record batch iterator from a boolean selection.
    pub fn get_record_batches_by_filter(
        &self,
        row_group_id: usize,
        selection: BooleanSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> BooleanSelectionIter {
        BooleanSelectionIter::new(
            self,
            row_group_id,
            selection,
            schema.clone(),
            parquet_column_ids.to_vec(),
            self.batch_size,
        )
    }

    /// Get a record batch from a row selection.
    pub fn get_record_batches<'a>(
        &'a self,
        row_group_id: usize,
        selection: BooleanSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> Box<dyn Iterator<Item = RecordBatch> + Send> {
        let selection_count = selection.row_count();
        let total_row_count = selection.len();
        let is_selective = (selection_count * 16) < total_row_count;
        let iter = ArrowArrayCache::get().get_record_batches_by_filter(
            row_group_id,
            selection,
            &schema,
            &parquet_column_ids,
        );
        let iter = if !is_selective {
            // means we have many small selections, we should coalesce them
            let coalesced = CoalescedIter::new(iter, self.batch_size);
            Box::new(coalesced) as Box<dyn Iterator<Item = RecordBatch> + Send>
        } else {
            Box::new(iter)
        };
        iter
    }

    /// Get a record batch iterator from a row selection.
    /// Each record batch is `take` (instead of `filter`) from the underlying RecordBatch iterator.
    pub fn get_record_batch_by_take(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> TakeRecordBatchIter {
        TakeRecordBatchIter::new(
            self,
            row_group_id,
            selection.clone().into(),
            schema.clone(),
            parquet_column_ids.to_vec(),
            self.batch_size,
        )
    }

    /// Get a record batch iterator from a row selection.
    pub fn get_record_batch_by_slice(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> SlicedRecordBatchIter {
        SlicedRecordBatchIter::new(
            self,
            row_group_id,
            selection.clone().into(),
            schema.clone(),
            parquet_column_ids.to_vec(),
            self.batch_size,
        )
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

        let column_cache = cache.entry(id.column_id).or_insert_with(AHashMap::new);

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

                if array.len() < self.batch_size {
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
pub struct SlicedRecordBatchIter<'a> {
    cache: &'a ArrowArrayCache,
    row_group_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    parquet_column_ids: Vec<usize>,
    row_id: usize,
    current_selected: usize,
    batch_size: usize,
}

impl<'a> SlicedRecordBatchIter<'a> {
    fn new(
        cache: &'a ArrowArrayCache,
        row_group_id: usize,
        selection: VecDeque<RowSelector>,
        mut schema: SchemaRef,
        parquet_column_ids: Vec<usize>,
        batch_size: usize,
    ) -> Self {
        if parquet_column_ids.is_empty() {
            let data_type = DataType::Struct(Fields::empty());
            let fields = vec![Field::new("empty", data_type, false)];
            schema = Arc::new(Schema::new(fields));
        }
        SlicedRecordBatchIter {
            cache,
            row_group_id,
            selection,
            schema,
            parquet_column_ids,
            row_id: 0,
            current_selected: 0,
            batch_size,
        }
    }

    /// Coalesces the output of the iterator into batches of a specified size.
    pub fn into_coalesced(self, batch_size: usize) -> CoalescedIter<Self> {
        CoalescedIter::new(self, batch_size)
    }
}

impl<'a> Iterator for SlicedRecordBatchIter<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(row_selector) = self.selection.pop_front() {
            if row_selector.skip {
                self.row_id += row_selector.row_count;
                continue;
            }

            assert!(self.current_selected < row_selector.row_count);
            let current_row_id = self.row_id + self.current_selected;
            let batch_row_id = (current_row_id / self.batch_size) * self.batch_size;
            let batch_row_count = self.batch_size - (current_row_id % self.batch_size);
            let offset = current_row_id.saturating_sub(batch_row_id);
            let want_to_select = std::cmp::min(
                batch_row_count,
                row_selector.row_count - self.current_selected,
            );

            let record_batch = if self.parquet_column_ids.is_empty() {
                make_dummy_record_batch(&self.schema, want_to_select)
            } else {
                let mut columns = Vec::with_capacity(self.parquet_column_ids.len());
                for &column_id in &self.parquet_column_ids {
                    let id = ArrayIdentifier::new(self.row_group_id, column_id, batch_row_id);
                    let mut array = self.cache.get_arrow_array(&id)?;
                    if offset > 0 || want_to_select < array.len() {
                        array = array.slice(offset, want_to_select);
                    }
                    columns.push(array);
                }

                assert_eq!(columns.len(), self.parquet_column_ids.len());
                RecordBatch::try_new(self.schema.clone(), columns).unwrap()
            };

            assert_eq!(record_batch.num_rows(), want_to_select);

            self.current_selected += want_to_select;
            if self.current_selected < row_selector.row_count {
                self.selection.push_front(row_selector);
            } else {
                self.row_id += row_selector.row_count;
                self.current_selected = 0;
            }
            return Some(record_batch);
        }
        None
    }
}

/// Iterator that yields coalesced `RecordBatch` items.
pub struct TakeRecordBatchIter<'a> {
    cache: &'a ArrowArrayCache,
    row_group_id: usize,
    selection: VecDeque<RowSelector>,
    schema: SchemaRef,
    projected_column_ids: Vec<usize>,
    row_id: usize,
    current_selected: usize,
    batch_size: usize,
    row_idx_of_current_batch: (usize, Vec<u32>), // (batch_id, row_idx)
}

impl<'a> TakeRecordBatchIter<'a> {
    fn new(
        cache: &'a ArrowArrayCache,
        row_group_id: usize,
        selection: VecDeque<RowSelector>,
        mut schema: SchemaRef,
        projected_column_ids: Vec<usize>,
        batch_size: usize,
    ) -> Self {
        let row_idx_of_current_batch = (0, vec![]);

        if projected_column_ids.is_empty() {
            let data_type = DataType::Struct(Fields::empty());
            let fields = vec![Field::new("empty", data_type, false)];
            schema = Arc::new(Schema::new(fields));
        }
        Self {
            cache,
            row_group_id,
            selection,
            schema,
            projected_column_ids,
            row_id: 0,
            current_selected: 0,
            batch_size,
            row_idx_of_current_batch,
        }
    }

    /// Coalesces the output of the iterator into batches of a specified size.
    pub fn into_coalesced(self, batch_size: usize) -> CoalescedIter<Self> {
        CoalescedIter::new(self, batch_size)
    }
}

fn make_dummy_record_batch(schema: &SchemaRef, len: usize) -> RecordBatch {
    let data_type = DataType::Struct(Fields::empty());
    let array = ArrayDataBuilder::new(data_type).len(len).build().unwrap();
    let array = StructArray::from(array);
    RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap()
}

impl<'a> Iterator for TakeRecordBatchIter<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(row_selector) = self.selection.pop_front() {
            if row_selector.skip {
                self.row_id += row_selector.row_count;
                continue;
            }

            for j in self.current_selected..row_selector.row_count {
                let new_row_idx = self.row_id + j;
                let new_batch_id = (new_row_idx / self.batch_size) * self.batch_size;

                if new_batch_id != self.row_idx_of_current_batch.0 {
                    if !self.row_idx_of_current_batch.1.is_empty() {
                        let batch_id = self.row_idx_of_current_batch.0;
                        let record_batch = if self.projected_column_ids.is_empty() {
                            // means we just make arrays with empty content but keep the row count.
                            make_dummy_record_batch(
                                &self.schema,
                                self.row_idx_of_current_batch.1.len(),
                            )
                        } else {
                            let mut columns = Vec::with_capacity(self.schema.fields().len());
                            let indices =
                                UInt32Array::from(self.row_idx_of_current_batch.1.clone());
                            for &column_id in &self.projected_column_ids {
                                let id =
                                    ArrayIdentifier::new(self.row_group_id, column_id, batch_id);
                                let mut array = self.cache.get_arrow_array(&id).unwrap();
                                array = arrow_select::take::take(&array, &indices, None).unwrap();
                                columns.push(array);
                            }

                            RecordBatch::try_new(self.schema.clone(), columns).unwrap()
                        };

                        self.row_idx_of_current_batch.1.clear();
                        self.row_idx_of_current_batch.0 = new_batch_id;
                        self.selection.push_front(row_selector);
                        self.current_selected = j;
                        return Some(record_batch);
                    }
                    self.row_idx_of_current_batch.0 = new_batch_id;
                }
                self.row_idx_of_current_batch
                    .1
                    .push((new_row_idx - self.row_idx_of_current_batch.0) as u32);
            }
            self.row_id += row_selector.row_count;
            self.current_selected = 0;
        }

        if !self.row_idx_of_current_batch.1.is_empty() {
            let batch_id = self.row_idx_of_current_batch.0;
            let record_batch = if self.projected_column_ids.is_empty() {
                make_dummy_record_batch(&self.schema, self.row_idx_of_current_batch.1.len())
            } else {
                let mut columns = Vec::with_capacity(self.projected_column_ids.len());
                let indices = UInt32Array::from(self.row_idx_of_current_batch.1.clone());
                for &column_id in &self.projected_column_ids {
                    let id = ArrayIdentifier::new(self.row_group_id, column_id, batch_id);
                    let mut array = self.cache.get_arrow_array(&id).unwrap();
                    array = arrow_select::take::take(&array, &indices, None).unwrap();
                    columns.push(array);
                }
                RecordBatch::try_new(self.schema.clone(), columns).unwrap()
            };

            self.row_idx_of_current_batch.1.clear();
            return Some(record_batch);
        }
        None
    }
}

/// CoalescedIter is an iterator that coalesces the output of an inner iterator into batches of a specified size.
pub struct CoalescedIter<T: Iterator<Item = RecordBatch> + Send> {
    inner: T,
    buffer: Vec<RecordBatch>,
    batch_size: usize,
}

impl<T: Iterator<Item = RecordBatch> + Send> CoalescedIter<T> {
    fn new(inner: T, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size,
            buffer: vec![],
        }
    }

    fn add_to_buffer(&mut self, record_batch: RecordBatch) -> Option<RecordBatch> {
        let existing_row_count = self
            .buffer
            .iter()
            .map(|batch| batch.num_rows())
            .sum::<usize>();
        if existing_row_count + record_batch.num_rows() < self.batch_size {
            self.buffer.push(record_batch);
            None
        } else {
            let schema = record_batch.schema();
            self.buffer.push(record_batch);
            let buffer = std::mem::replace(&mut self.buffer, vec![]);
            let coalesced = concat_batches(&schema, &buffer).unwrap();
            Some(coalesced)
        }
    }
}

impl<T: Iterator<Item = RecordBatch> + Send> Iterator for CoalescedIter<T> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(batch) = self.inner.next() {
            if let Some(coalesced) = self.add_to_buffer(batch) {
                return Some(coalesced);
            }
        }
        if !self.buffer.is_empty() {
            let coalesced =
                concat_batches(&self.buffer.first().unwrap().schema(), &self.buffer).unwrap();
            self.buffer.clear();
            return Some(coalesced);
        }
        None
    }
}

/// BooleanSelectionIter is an iterator that yields record batches based on a boolean selection.
pub struct BooleanSelectionIter<'a> {
    cache: &'a ArrowArrayCache,
    selection: BooleanSelection,
    row_group_id: usize,
    schema: SchemaRef,
    parquet_column_ids: Vec<usize>,
    batch_size: usize,
    cur_row_id: usize,
}

impl<'a> BooleanSelectionIter<'a> {
    fn new(
        cache: &'a ArrowArrayCache,
        row_group_id: usize,
        selection: BooleanSelection,
        schema: SchemaRef,
        parquet_column_ids: Vec<usize>,
        batch_size: usize,
    ) -> Self {
        Self {
            cache,
            selection,
            row_group_id,
            schema,
            parquet_column_ids,
            batch_size,
            cur_row_id: 0,
        }
    }

    /// Coalesces the output of the iterator into batches of a specified size.
    pub fn into_coalesced(self, batch_size: usize) -> CoalescedIter<Self> {
        CoalescedIter::new(self, batch_size)
    }
}

impl<'a> Iterator for BooleanSelectionIter<'a> {
    type Item = RecordBatch;
    fn next(&mut self) -> Option<Self::Item> {
        while self.cur_row_id < self.selection.len() {
            let want_to_select = self.batch_size.min(self.selection.len() - self.cur_row_id);
            let selection = self.selection.slice(self.cur_row_id, want_to_select);
            if selection.true_count() == 0 {
                // no rows are selected, skip this batch
                self.cur_row_id += want_to_select;
                continue;
            }

            let record_batch = if self.parquet_column_ids.is_empty() {
                make_dummy_record_batch(&self.schema, want_to_select)
            } else {
                let mut columns = Vec::with_capacity(self.schema.fields().len());
                for &column_id in &self.parquet_column_ids {
                    let id = ArrayIdentifier::new(self.row_group_id, column_id, self.cur_row_id);
                    let mut array = self.cache.get_arrow_array(&id).unwrap();
                    array = arrow_select::filter::filter(&array, &selection).unwrap();
                    columns.push(array);
                }
                RecordBatch::try_new(self.schema.clone(), columns).unwrap()
            };

            self.cur_row_id += want_to_select;
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
fn intersect_ranges(
    mut base: HashSet<Range<usize>>,
    input: &HashSet<Range<usize>>,
) -> HashSet<Range<usize>> {
    base.retain(|range| input.contains(range));
    base
}

#[cfg(test)]
mod tests {
    use crate::arrow::arrow_reader::BooleanSelection;

    use super::*;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn set_up_cache() -> ArrowArrayCache {
        /// Helper function to create a RecordBatch with a single Int32 column.
        fn create_record_batch(name: &str, num_rows: usize, start: i32) -> RecordBatch {
            let array = Int32Array::from_iter_values(start..(start + num_rows as i32));
            let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
            RecordBatch::try_new(schema, vec![Arc::new(array) as ArrayRef]).unwrap()
        }

        let cache = ArrowArrayCache::new(ArrowCacheMode::InMemory, 32);

        let row_group_id = 0;
        let column_id = 0;

        // Populate the cache with 42 rows of data split into two batches
        // Batch 1: rows 0-31
        let batch1 = create_record_batch("a", 32, 0);
        let id1 = ArrayIdentifier::new(row_group_id, column_id, 0);
        cache.insert_arrow_array(&id1, batch1.column(0).clone());

        // Batch 2: rows 32-41
        let batch2 = create_record_batch("a", 10, 32);
        let id2 = ArrayIdentifier::new(row_group_id, column_id, 32);
        cache.insert_arrow_array(&id2, batch2.column(0).clone());
        cache
    }

    #[test]
    fn test_get_coalesced_record_batch_iter() {
        let cache = set_up_cache();
        let row_group_id = 0;
        let column_id = 0;
        let parquet_column_ids = vec![column_id];
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Define various row selections
        let selections = gen_selections();

        let expected: Vec<Vec<Vec<i32>>> = vec![
            vec![(0..32).collect(), (32..42).collect()],
            vec![(0..32).collect()],
            vec![(33..38).collect()],
            vec![(0..8).chain(10..26).chain(36..40).collect()],
            vec![(8..10).chain(26..32).chain(32..36).collect()],
        ];

        for (selection, expected) in selections.iter().zip(expected) {
            let expected = expected
                .into_iter()
                .map(|range| Int32Array::from(range))
                .collect::<Vec<_>>();
            let selection = RowSelection::from(selection.clone());
            let by_take_record_batches = cache
                .get_record_batch_by_take(row_group_id, &selection, &schema, &parquet_column_ids)
                .into_coalesced(32)
                .collect::<Vec<_>>();
            check_result(by_take_record_batches, &expected);

            let by_slice_record_batches = cache
                .get_record_batch_by_slice(row_group_id, &selection, &schema, &parquet_column_ids)
                .into_coalesced(32)
                .collect::<Vec<_>>();
            check_result(by_slice_record_batches, &expected);

            let by_filter_record_batches = cache
                .get_record_batches_by_filter(
                    row_group_id,
                    BooleanSelection::from(selection),
                    &schema,
                    &parquet_column_ids,
                )
                .into_coalesced(32)
                .collect::<Vec<_>>();
            check_result(by_filter_record_batches, &expected);
        }
    }

    #[test]
    fn test_get_coalesced_record_batch_iter_no_column() {
        let cache = set_up_cache();
        let row_group_id = 0;
        let parquet_column_ids = vec![];
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        // Define various row selections
        let selections: Vec<(Vec<RowSelector>, Vec<Vec<i32>>)> = vec![
            (
                vec![RowSelector::select(42)],
                vec![(0..32).collect(), (32..42).collect()],
            ),
            (
                vec![RowSelector::select(15), RowSelector::select(10)],
                vec![(0..25).collect()],
            ),
            (
                vec![
                    RowSelector::skip(33),
                    RowSelector::select(5),
                    RowSelector::select(3),
                ],
                vec![(33..41).collect()],
            ),
            (
                vec![RowSelector::skip(16), RowSelector::select(22)],
                vec![(16..38).collect()],
            ),
            (
                vec![
                    RowSelector::select(16),
                    RowSelector::skip(20),
                    RowSelector::select(4),
                    RowSelector::select(2),
                ],
                vec![(0..16).into_iter().chain(36..42).collect()],
            ),
        ];

        for (selection, expected) in selections.iter() {
            let selection = RowSelection::from(selection.clone());
            let record_batches = cache
                .get_record_batch_by_take(row_group_id, &selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            assert_eq!(record_batches.len(), expected.len());
            for (batch, expected) in record_batches.into_iter().zip(expected) {
                assert_eq!(batch.num_rows(), expected.len());
            }
        }
    }

    #[test]
    fn test_get_record_batch() {
        let row_group_id = 0;
        let column_id = 0;
        let parquet_column_ids = vec![column_id];
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));

        let cache = set_up_cache();

        // Define various row selections
        let selections = gen_selections();

        let by_slice_expected: Vec<Vec<Vec<i32>>> = vec![
            vec![(0..32).collect(), (32..42).collect()],
            vec![(0..32).collect()],
            vec![(33..38).collect()],
            vec![(0..8).collect(), (10..26).collect(), (36..40).collect()],
            vec![(8..10).collect(), (26..32).collect(), (32..36).collect()],
        ];

        for (selection, expected) in selections.iter().zip(by_slice_expected) {
            let expected = expected
                .into_iter()
                .map(|range| Int32Array::from(range))
                .collect::<Vec<_>>();
            let selection = RowSelection::from(selection.clone());
            let record_batches = cache
                .get_record_batch_by_slice(row_group_id, &selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            check_result(record_batches, &expected);
        }

        let by_take_filter_expected: Vec<Vec<Vec<i32>>> = vec![
            vec![(0..32).collect(), (32..42).collect()],
            vec![(0..32).collect()],
            vec![(33..38).collect()],
            vec![(0..8).chain(10..26).collect(), (36..40).collect()],
            vec![(8..10).chain(26..32).collect(), (32..36).collect()],
        ];

        for (selection, expected) in selections.iter().zip(by_take_filter_expected.iter()) {
            let expected = expected
                .into_iter()
                .map(|range| Int32Array::from(range.clone()))
                .collect::<Vec<_>>();

            let selection = RowSelection::from(selection.clone());
            let take_record_batches = cache
                .get_record_batch_by_take(row_group_id, &selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            check_result(take_record_batches, &expected);

            let selection = BooleanSelection::from(selection);
            let record_batches = cache
                .get_record_batches_by_filter(row_group_id, selection, &schema, &parquet_column_ids)
                .collect::<Vec<_>>();
            check_result(record_batches, &expected);
        }
    }

    fn gen_selections() -> Vec<Vec<RowSelector>> {
        vec![
            vec![RowSelector::select(42)],
            vec![RowSelector::select(32)],
            vec![RowSelector::skip(33), RowSelector::select(5)],
            vec![
                RowSelector::select(8),
                RowSelector::skip(2),
                RowSelector::select(16),
                RowSelector::skip(10),
                RowSelector::select(4),
            ],
            vec![
                RowSelector::skip(8),
                RowSelector::select(2),
                RowSelector::skip(16),
                RowSelector::select(10),
                RowSelector::skip(4),
            ],
        ]
    }

    fn check_result(record_batches: Vec<RecordBatch>, expected: &Vec<Int32Array>) {
        assert_eq!(record_batches.len(), expected.len());
        for (batch, expected) in record_batches.into_iter().zip(expected) {
            let actual = batch.column(0).as_primitive::<ArrowInt32Type>();
            assert_eq!(actual, expected);
        }
    }
}
