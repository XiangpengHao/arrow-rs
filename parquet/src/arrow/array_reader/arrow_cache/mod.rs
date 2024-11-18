use crate::arrow::arrow_reader::{ArrowPredicate, BooleanSelection, RowSelection};
use ahash::AHashMap;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use etc_array::{AsEtcArray, EtcArrayRef, EtcPrimitiveArray, EtcStringArray, EtcStringMetadata};
use lock_spec::{
    LockBefore, LockColumnMapping, LockCtx, LockDiskFile, LockEtcCompressorMetadata,
    LockEtcFsstCompressor, LockVortexCompression, LockedEntry, OrderedMutex, OrderedRwLock,
};
use utils::RangedFile;
use vortex::arrow::FromArrowArray;
use vortex::IntoCanonical;
use vortex_sampling_compressor::compressors::alp::ALPCompressor;
use vortex_sampling_compressor::compressors::bitpacked::BITPACK_WITH_PATCHES;
use vortex_sampling_compressor::compressors::delta::DeltaCompressor;
use vortex_sampling_compressor::compressors::dict::DictCompressor;
use vortex_sampling_compressor::compressors::fsst::FSSTCompressor;
use vortex_sampling_compressor::compressors::r#for::FoRCompressor;
use vortex_sampling_compressor::compressors::{CompressionTree, CompressorRef};
use vortex_sampling_compressor::SamplingCompressor;

/// An array that stores strings in a dictionary format, with a bit-packed array for the keys and a FSST array for the values.
pub mod etc_array;
mod iter;
mod lock_spec;
mod stats;
mod utils;

pub use stats::ArrowCacheStatistics;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type as ArrowDate32Type, Date64Type as ArrowDate64Type, Int16Type as ArrowInt16Type,
    Int32Type as ArrowInt32Type, Int64Type as ArrowInt64Type, Int8Type as ArrowInt8Type,
    UInt16Type as ArrowUInt16Type, UInt32Type as ArrowUInt32Type, UInt64Type as ArrowUInt64Type,
    UInt8Type as ArrowUInt8Type,
};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, RecordBatchWriter};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use std::collections::HashSet;
use std::fmt::Display;
use std::io::Seek;
use std::ops::{DerefMut, Range};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, RwLockReadGuard};

static ARROW_ARRAY_CACHE: LazyLock<ArrowArrayCache> =
    LazyLock::new(ArrowArrayCache::initialize_from_env);

static ARROW_DISK_CACHE_PATH: &str = "target/arrow_disk_cache.etc";

/// Row offset -> (Arrow Array, hit count)
type Rows = AHashMap<usize, CachedEntry>;

/// Column offset -> RowMapping
type Columns = AHashMap<usize, Rows>;

#[derive(Debug)]
enum CacheMode {
    InMemory,
    OnDisk(OrderedMutex<LockDiskFile, std::fs::File>),
    NoCache,
    Vortex(CompressorStates),
    Etc(EtcCompressorStates),
}

#[derive(Debug)]
struct EtcCompressorStates {
    #[allow(dead_code)]
    metadata:
        OrderedRwLock<LockEtcCompressorMetadata, AHashMap<ArrayIdentifier, EtcStringMetadata>>,
    fsst_compressor:
        OrderedRwLock<LockEtcFsstCompressor, AHashMap<(usize, usize), Arc<fsst::Compressor>>>, // (row_group_id, column_id) -> compressor
}

impl EtcCompressorStates {
    fn new() -> Self {
        Self {
            metadata: OrderedRwLock::new(AHashMap::new()),
            fsst_compressor: OrderedRwLock::new(AHashMap::new()),
        }
    }
}

#[derive(Debug)]
struct CompressorStates {
    compressor: SamplingCompressor<'static>,
    compress_tree:
        OrderedRwLock<LockVortexCompression, AHashMap<(usize, usize), CompressionTree<'static>>>,
}

impl CompressorStates {
    fn new() -> Self {
        let compressor = SamplingCompressor::new_with_options(
            HashSet::from([
                &ALPCompressor as CompressorRef,
                &BITPACK_WITH_PATCHES,
                &DeltaCompressor,
                &FSSTCompressor,
                &DictCompressor,
                &FoRCompressor,
            ]),
            Default::default(),
        );
        Self {
            compressor,
            compress_tree: OrderedRwLock::new(AHashMap::new()),
        }
    }

    fn compress<L>(
        &self,
        array: &vortex::Array,
        row_group_id: usize,
        column_id: usize,
        ctx: &mut LockCtx<L>,
    ) -> Arc<vortex::Array>
    where
        L: LockBefore<LockVortexCompression>,
    {
        let (state_lock, _) = self.compress_tree.read(ctx);

        if let Some(tree) = state_lock.get(&(row_group_id, column_id)) {
            let compressed = self.compressor.compress(array, Some(tree)).unwrap();
            return Arc::new(compressed.into_array());
        }
        drop(state_lock);

        let compressed = self.compressor.compress(array, None).unwrap();
        let (array, tree) = compressed.into_parts();
        if let Some(tree) = tree {
            let (mut state_lock, _) = self.compress_tree.write(ctx);
            state_lock.insert((row_group_id, column_id), tree);
        }
        Arc::new(array)
    }
}

struct CachedEntryInner {
    value: CachedValue,
    row_count: u32,
    hit_count: AtomicU32,
}

impl CachedEntryInner {
    fn new(value: CachedValue, row_count: u32) -> Self {
        Self {
            value,
            row_count,
            hit_count: AtomicU32::new(0),
        }
    }
}

struct CachedEntry {
    inner: OrderedRwLock<LockedEntry, CachedEntryInner>,
}

impl CachedEntry {
    fn row_count<L>(&self, ctx: &mut LockCtx<L>) -> u32
    where
        L: LockBefore<LockedEntry>,
    {
        self.inner.read(ctx).0.row_count
    }

    fn increment_hit_count<L>(&self, ctx: &mut LockCtx<L>)
    where
        L: LockBefore<LockedEntry>,
    {
        self.inner
            .read(ctx)
            .0
            .hit_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn value<L>(
        &self,
        ctx: &mut LockCtx<L>,
    ) -> (RwLockReadGuard<'_, CachedEntryInner>, LockCtx<LockedEntry>)
    where
        L: LockBefore<LockedEntry>,
    {
        self.inner.read(ctx)
    }

    fn new_in_memory(array: ArrayRef) -> Self {
        let len = array.len();
        let val = CachedValue::ArrowMemory(array);
        CachedEntry {
            inner: OrderedRwLock::new(CachedEntryInner::new(val, len as u32)),
        }
    }

    fn new(value: CachedValue, row_count: usize) -> Self {
        CachedEntry {
            inner: OrderedRwLock::new(CachedEntryInner::new(value, row_count as u32)),
        }
    }

    fn new_vortex(array: Arc<vortex::Array>) -> Self {
        let len = array.len();
        let val = CachedValue::Vortex(array);
        CachedEntry {
            inner: OrderedRwLock::new(CachedEntryInner::new(val, len as u32)),
        }
    }
}

enum CachedValue {
    ArrowMemory(ArrayRef),
    Vortex(Arc<vortex::Array>),
    ArrowDisk(Range<u64>),
    Etc(EtcArrayRef),
}

impl CachedValue {
    fn memory_usage(&self) -> usize {
        match self {
            Self::ArrowMemory(array) => array.get_array_memory_size(),
            Self::ArrowDisk(_) => 0,
            Self::Vortex(array) => array.nbytes(),
            Self::Etc(array) => array.get_array_memory_size(),
        }
    }

    fn convert_to(&mut self, to: &CacheMode, ctx: &mut LockCtx<LockColumnMapping>) {
        match (&self, to) {
            (Self::ArrowMemory(v), CacheMode::OnDisk(file)) => {
                let (mut file, _) = file.lock(ctx);

                // Align start_pos to next 512 boundary for better disk I/O
                let start_pos = file.metadata().unwrap().len();
                let start_pos = (start_pos + 511) & !511;
                let start_pos = file.seek(std::io::SeekFrom::Start(start_pos)).unwrap();

                let mut writer = std::io::BufWriter::new(file.deref_mut());
                let schema = Arc::new(Schema::new(vec![Field::new(
                    "_",
                    v.data_type().clone(),
                    v.is_nullable(),
                )]));
                let mut arrow_writer = FileWriter::try_new(&mut writer, &schema).unwrap();
                let record_batch = RecordBatch::try_new(schema, vec![v.clone()]).unwrap();
                arrow_writer.write(&record_batch).unwrap();
                arrow_writer.close().unwrap();

                let file = writer.into_inner().unwrap();
                let end_pos = file.stream_position().unwrap();
                *self = CachedValue::ArrowDisk(start_pos..end_pos);
            }
            _ => unimplemented!("convert {} to {:?} not implemented", self, to),
        }
    }
}

impl Display for CachedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArrowMemory(_) => write!(f, "ArrowMemory"),
            Self::Vortex(_) => write!(f, "Vortex"),
            Self::ArrowDisk(_) => write!(f, "ArrowDisk"),
            Self::Etc(_) => write!(f, "Etc"),
        }
    }
}

/// ArrayIdentifier is used to identify an array in the cache.
#[derive(Debug)]
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

/// CacheType is used to identify the type of cache.
#[derive(Debug, serde::Serialize)]
pub enum CacheType {
    /// InMemory cache
    InMemory,
    /// OnDisk cache
    OnDisk,
    /// Vortex cache
    Vortex,
    /// Etc cache
    Etc,
}

/// ArrowArrayCache is used to cache arrow arrays in memory, on disk, or in a vortex.
pub struct ArrowArrayCache {
    /// Vec of RwLocks, where index is the row group index and value is the ColumnMapping
    value: Vec<OrderedRwLock<LockColumnMapping, Columns>>,
    cache_mode: CacheMode,
    batch_size: usize,
}

impl ArrowArrayCache {
    fn initialize_from_env() -> Self {
        let cache_mode = std::env::var("ARROW_CACHE_MODE").map_or(CacheMode::NoCache, |v| match v
            .to_lowercase()
            .as_str()
        {
            "disk" => CacheMode::OnDisk(OrderedMutex::new(
                std::fs::File::create(ARROW_DISK_CACHE_PATH).unwrap(),
            )),
            "inmemory" => CacheMode::InMemory,
            "nocache" => CacheMode::NoCache,
            "vortex" => CacheMode::Vortex(CompressorStates::new()),
            "etc" => CacheMode::Etc(EtcCompressorStates::new()),
            _ => panic!(
                "Invalid cache mode: {}, must be one of [disk, inmemory, nocache, vortex, etc]",
                v
            ),
        });

        println!("Initializing ArrowArrayCache with {:?}", cache_mode);

        ArrowArrayCache::new(cache_mode, 8192)
    }

    /// Create a new ArrowArrayCache.
    fn new(cache_mode: CacheMode, batch_size: usize) -> Self {
        assert!(batch_size.is_power_of_two());
        const MAX_ROW_GROUPS: usize = 512;
        ArrowArrayCache {
            value: (0..MAX_ROW_GROUPS)
                .map(|_| OrderedRwLock::new(Columns::new()))
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
        !matches!(self.cache_mode, CacheMode::NoCache)
    }

    /// Reset the cache.
    pub fn reset(&self) {
        let mut ctx = LockCtx::UNLOCKED;
        for row_group in self.value.iter() {
            let (mut row_group, _) = row_group.write(&mut ctx);
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
        let mut ctx = LockCtx::UNLOCKED;
        let (v, mut ctx) = self.value[row_group_id].read(&mut ctx);
        let rows = v.get(&column_id)?;

        let mut result_ranges = HashSet::new();
        for (row_id, cached_entry) in rows.iter() {
            let start = *row_id;
            let end = row_id + cached_entry.row_count(&mut ctx) as usize;

            result_ranges.insert(start..end);
        }
        Some(result_ranges)
    }

    /// Get a record batch iterator from a boolean selection.
    pub fn get_record_batches_by_filter<'a>(
        &'a self,
        row_group_id: usize,
        selection: BooleanSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> iter::BooleanSelectionIter<'a> {
        iter::BooleanSelectionIter::new(
            self,
            row_group_id,
            selection,
            schema.clone(),
            parquet_column_ids.to_vec(),
            self.batch_size,
        )
    }

    /// Get a record batch iterator from a boolean selection with a predicate.
    pub fn get_record_batches_by_selection_with_predicate<'a, 'b>(
        &'a self,
        row_group_id: usize,
        selection: &'b BooleanSelection,
        schema: &SchemaRef,
        parquet_column_id: usize,
        predicate: &'b mut Box<dyn ArrowPredicate>,
    ) -> iter::BooleanSelectionPredicateIter<'a, 'b> {
        iter::BooleanSelectionPredicateIter::new(
            self,
            row_group_id,
            selection,
            schema.clone(),
            parquet_column_id,
            self.batch_size,
            predicate,
        )
    }

    /// Get a record batch from a row selection.
    pub fn get_record_batches(
        &self,
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
            schema,
            parquet_column_ids,
        );
        if !is_selective {
            // means we have many small selections, we should coalesce them
            let coalesced = iter::CoalescedIter::new(iter, self.batch_size);
            Box::new(coalesced) as Box<dyn Iterator<Item = RecordBatch> + Send>
        } else {
            Box::new(iter)
        }
    }

    /// Get a record batch iterator from a row selection.
    /// Each record batch is `take` (instead of `filter`) from the underlying RecordBatch iterator.
    pub fn get_record_batch_by_take(
        &self,
        row_group_id: usize,
        selection: &RowSelection,
        schema: &SchemaRef,
        parquet_column_ids: &[usize],
    ) -> iter::TakeRecordBatchIter {
        iter::TakeRecordBatchIter::new(
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
    ) -> iter::SlicedRecordBatchIter {
        iter::SlicedRecordBatchIter::new(
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

    /// Get an arrow array from the cache with a selection.
    pub fn get_arrow_array_with_selection_and_predicate(
        &self,
        id: &ArrayIdentifier,
        selection: Option<&BooleanArray>,
        predicate: &mut Box<dyn ArrowPredicate>,
        schema: &SchemaRef,
    ) -> Option<BooleanArray> {
        if matches!(self.cache_mode, CacheMode::NoCache) {
            return None;
        }

        let mut ctx = LockCtx::UNLOCKED;
        let (cache, mut ctx) = self.value[id.row_group_id].read(&mut ctx);

        let column_cache = cache.get(&id.column_id).unwrap();
        let cached_entry = column_cache.get(&id.row_id).unwrap();
        cached_entry.increment_hit_count(&mut ctx);
        let cached_entry = cached_entry.value(&mut ctx);
        match &cached_entry.0.value {
            CachedValue::ArrowMemory(array) => {
                let array = match selection {
                    Some(selection) => arrow_select::filter::filter(array, selection).unwrap(),
                    None => array.clone(),
                };
                let record_batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
                let array = predicate.evaluate(record_batch).unwrap();
                Some(array)
            }
            CachedValue::ArrowDisk(range) => {
                let file = std::fs::File::open(ARROW_DISK_CACHE_PATH).unwrap();
                let ranged_file = RangedFile::new(file, range.clone()).unwrap();
                let reader = std::io::BufReader::new(ranged_file);

                let mut arrow_reader = FileReader::try_new(reader, None).unwrap();
                let batch = arrow_reader.next().unwrap().unwrap();
                let batch = match selection {
                    Some(selection) => {
                        arrow_select::filter::filter_record_batch(&batch, selection).unwrap()
                    }
                    None => batch,
                };
                let array = predicate.evaluate(batch).unwrap();
                Some(array)
            }
            CachedValue::Vortex(array) => match selection {
                Some(selection) => {
                    let selection = vortex::Array::from_arrow(selection, false);
                    let filtered = vortex::compute::filter(array, &selection).unwrap();
                    let array = predicate.evaluate_any(&filtered).unwrap();
                    Some(array)
                }
                None => Some(predicate.evaluate_any(array).unwrap()),
            },
            CachedValue::Etc(array) => match selection {
                Some(selection) => {
                    if let Some(_string_array) = array.as_string_array_opt() {
                        let filtered = array.filter(selection);
                        let result = predicate.evaluate_any(&filtered).unwrap();
                        Some(result)
                    } else {
                        let etc_primitive = array.to_arrow_array();

                        let filtered =
                            arrow_select::filter::filter(&etc_primitive, selection).unwrap();

                        let schema = Schema::new(vec![Field::new(
                            "",
                            etc_primitive.data_type().clone(),
                            etc_primitive.is_nullable(),
                        )]);
                        let record_batch =
                            RecordBatch::try_new(Arc::new(schema), vec![filtered]).unwrap();
                        let result = predicate.evaluate(record_batch).unwrap();
                        Some(result)
                    }
                }
                None => Some(predicate.evaluate_any(array).unwrap()),
            },
        }
    }

    /// Get an arrow array from the cache with a selection.
    pub fn get_arrow_array_with_selection(
        &self,
        id: &ArrayIdentifier,
        selection: Option<&BooleanArray>,
    ) -> Option<ArrayRef> {
        if matches!(self.cache_mode, CacheMode::NoCache) {
            return None;
        }

        let mut ctx = LockCtx::UNLOCKED;

        let (cache, mut ctx_column) = self.value[id.row_group_id].read(&mut ctx);

        let column_cache = cache.get(&id.column_id)?;
        let cached_entry = column_cache.get(&id.row_id)?;
        cached_entry.increment_hit_count(&mut ctx_column);
        let cached_entry = cached_entry.value(&mut ctx_column);
        match &cached_entry.0.value {
            CachedValue::ArrowMemory(array) => match selection {
                Some(selection) => Some(arrow_select::filter::filter(array, selection).unwrap()),
                None => Some(array.clone()),
            },
            CachedValue::ArrowDisk(range) => {
                let file = std::fs::File::open(ARROW_DISK_CACHE_PATH).ok()?;
                let ranged_file = RangedFile::new(file, range.clone()).ok()?;

                let reader = std::io::BufReader::new(ranged_file);
                let mut arrow_reader = FileReader::try_new(reader, None).ok()?;
                let batch = arrow_reader.next().unwrap().unwrap();
                let array = batch.column(0);
                match selection {
                    Some(selection) => {
                        Some(arrow_select::filter::filter(array, selection).unwrap())
                    }
                    None => Some(array.clone()),
                }
            }
            CachedValue::Vortex(array) => match selection {
                Some(selection) => {
                    let predicate = vortex::Array::from_arrow(selection, false);
                    let filtered = vortex::compute::filter(array, &predicate).unwrap();
                    let array = filtered.into_canonical().unwrap();
                    let canonical_array = array.into_arrow().unwrap();

                    // let canonical_array = (**array)
                    //     .clone()
                    //     .into_canonical()
                    //     .unwrap()
                    //     .into_arrow()
                    //     .unwrap();
                    // let canonical_array =
                    //     arrow_select::filter::filter(&canonical_array, selection).unwrap();
                    Some(canonical_array)
                }
                None => Some(
                    (**array)
                        .clone()
                        .into_canonical()
                        .unwrap()
                        .into_arrow()
                        .unwrap(),
                ),
            },
            CachedValue::Etc(array) => match selection {
                Some(selection) => {
                    if let Some(string_array) = array.as_string_array_opt() {
                        let arrow_array = string_array.to_dict_string_with_selection(selection);
                        Some(Arc::new(arrow_array))
                    } else {
                        let etc_primitive = array.to_arrow_array();
                        let filtered =
                            arrow_select::filter::filter(&etc_primitive, selection).unwrap();
                        Some(filtered)
                    }
                }
                None => {
                    if let Some(string_array) = array.as_string_array_opt() {
                        let arrow_array = string_array.to_dict_string();
                        Some(Arc::new(arrow_array))
                    } else {
                        let arrow_array = array.to_arrow_array();
                        Some(arrow_array)
                    }
                }
            },
        }
    }

    /// Get an arrow array from the cache.
    pub fn get_arrow_array(&self, id: &ArrayIdentifier) -> Option<ArrayRef> {
        self.get_arrow_array_with_selection(id, None)
    }

    pub(crate) fn is_cached<L>(&self, id: &ArrayIdentifier, ctx: &mut LockCtx<L>) -> bool
    where
        L: LockBefore<LockColumnMapping>,
    {
        let (cache, _) = self.value[id.row_group_id].read(ctx);
        cache.contains_key(&id.column_id)
            && cache.get(&id.column_id).unwrap().contains_key(&id.row_id)
    }

    pub(crate) fn get_len(&self, id: &ArrayIdentifier) -> Option<usize> {
        let mut ctx = LockCtx::UNLOCKED;
        let (cache, mut ctx) = self.value[id.row_group_id].read(&mut ctx);
        let column_cache = cache.get(&id.column_id)?;
        let cached_entry = column_cache.get(&id.row_id)?;
        Some(cached_entry.row_count(&mut ctx) as usize)
    }

    /// Insert an arrow array into the cache.
    pub(crate) fn insert_arrow_array(&self, id: &ArrayIdentifier, array: ArrayRef) {
        if matches!(self.cache_mode, CacheMode::NoCache) {
            return;
        }

        let mut ctx = LockCtx::UNLOCKED;
        if self.is_cached(id, &mut ctx) {
            return;
        }

        let (mut cache, mut column_ctx) = self.value[id.row_group_id].write(&mut ctx);

        let column_cache = cache.entry(id.column_id).or_insert_with(AHashMap::new);

        match &self.cache_mode {
            CacheMode::InMemory => {
                let old = column_cache.insert(id.row_id, CachedEntry::new_in_memory(array));
                assert!(old.is_none());
            }
            CacheMode::OnDisk(_file) => {
                let row_count = array.len();
                let mut cached_value = CachedValue::ArrowMemory(array);
                cached_value.convert_to(&self.cache_mode, &mut column_ctx);

                column_cache.insert(id.row_id, CachedEntry::new(cached_value, row_count));
            }
            CacheMode::Vortex(compressor) => {
                let data_type = array.data_type();

                if array.len() < self.batch_size {
                    // our batch is too small.
                    column_cache.insert(id.row_id, CachedEntry::new_in_memory(array));
                    return;
                }

                drop(cache); // blocking call below, release the lock

                let uncompressed = match data_type {
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
                    DataType::Utf8View => {
                        let string_array = array.as_string_view();
                        vortex::Array::from_arrow(string_array, array.logical_nulls().is_some())
                    }
                    _ => {
                        unimplemented!("data type {:?} not implemented", data_type);
                    }
                };

                let compressed = compressor.compress(
                    &uncompressed,
                    id.row_group_id,
                    id.column_id,
                    &mut column_ctx,
                );

                let (mut cache, _ctx) = self.value[id.row_group_id].write(&mut ctx);
                let column_cache = cache.entry(id.column_id).or_insert_with(AHashMap::new);
                column_cache.insert(id.row_id, CachedEntry::new_vortex(compressed));
            }

            CacheMode::NoCache => {
                unreachable!()
            }

            CacheMode::Etc(ref states) => {
                let data_type = array.data_type();
                let array = array.as_ref();
                if data_type.is_primitive() {
                    let primitive: EtcArrayRef = match data_type {
                        DataType::Int8 => {
                            Arc::new(EtcPrimitiveArray::<ArrowInt8Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt8Type>().clone(),
                            ))
                        }
                        DataType::Int16 => {
                            Arc::new(EtcPrimitiveArray::<ArrowInt16Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt16Type>().clone(),
                            ))
                        }
                        DataType::Int32 => {
                            Arc::new(EtcPrimitiveArray::<ArrowInt32Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt32Type>().clone(),
                            ))
                        }
                        DataType::Int64 => {
                            Arc::new(EtcPrimitiveArray::<ArrowInt64Type>::from_arrow_array(
                                array.as_primitive::<ArrowInt64Type>().clone(),
                            ))
                        }
                        DataType::UInt8 => {
                            Arc::new(EtcPrimitiveArray::<ArrowUInt8Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt8Type>().clone(),
                            ))
                        }
                        DataType::UInt16 => {
                            Arc::new(EtcPrimitiveArray::<ArrowUInt16Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt16Type>().clone(),
                            ))
                        }
                        DataType::UInt32 => {
                            Arc::new(EtcPrimitiveArray::<ArrowUInt32Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt32Type>().clone(),
                            ))
                        }
                        DataType::UInt64 => {
                            Arc::new(EtcPrimitiveArray::<ArrowUInt64Type>::from_arrow_array(
                                array.as_primitive::<ArrowUInt64Type>().clone(),
                            ))
                        }
                        _ => panic!("unsupported data type {:?}", data_type),
                    };
                    column_cache.insert(
                        id.row_id,
                        CachedEntry::new(CachedValue::Etc(primitive), array.len()),
                    );
                    return;
                }
                // other types
                match array.data_type() {
                    DataType::Utf8 => {
                        let (compressor, _) = states.fsst_compressor.read(&mut column_ctx);
                        if let Some(compressor) = compressor.get(&(id.row_group_id, id.column_id)) {
                            let compressed = EtcStringArray::from_string_array(
                                array.as_string::<i32>(),
                                Some(compressor.clone()),
                            );
                            column_cache.insert(
                                id.row_id,
                                CachedEntry::new(
                                    CachedValue::Etc(Arc::new(compressed)),
                                    array.len(),
                                ),
                            );
                            return;
                        }

                        drop(compressor);
                        let (mut compressors, _) = states.fsst_compressor.write(&mut column_ctx);
                        let compressed =
                            EtcStringArray::from_string_array(array.as_string::<i32>(), None);
                        let compressor = compressed.compressor();
                        compressors.insert((id.row_group_id, id.column_id), compressor);
                        column_cache.insert(
                            id.row_id,
                            CachedEntry::new(CachedValue::Etc(Arc::new(compressed)), array.len()),
                        );
                    }
                    DataType::Dictionary(_, _) => {
                        if let Some(dict_array) = array.as_dictionary_opt::<ArrowUInt16Type>() {
                            let values = dict_array.values();
                            if let Some(_string_array) = values.as_string_opt::<i32>() {
                                let etc_array = EtcStringArray::from_dict_array(&dict_array);
                                column_cache.insert(
                                    id.row_id,
                                    CachedEntry::new(
                                        CachedValue::Etc(Arc::new(etc_array)),
                                        array.len(),
                                    ),
                                );
                                return;
                            }
                        }

                        panic!("unsupported data type {:?}", array.data_type());
                    }
                    _ => panic!("unsupported data type {:?}", array.data_type()),
                }
            }
        }
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
