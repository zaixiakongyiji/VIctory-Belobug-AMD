//! Host memory for the textures the title fills, and for their staging
//! buffers, cut from blocks made once.
//!
//! Each such texture had an allocation of its own and its staging buffer
//! another. On an RX 7900 XTX an allocation takes about 310 microseconds
//! whatever its size, on one thread (`texture::allocation_costs`), and a
//! texture cut from a block 6: the pass that makes the textures shaders
//! pick by index (`draw.rs` `heap_textures`) got through 350-500 of the
//! 24,000 queued when the world loads in each of its 250 ms turns, some
//! forty seconds of a world without its textures.
//! `VICTORY_TEXTURE_POOL=0` makes every one an allocation of its own again.
//!
//! A piece given back is used again only two seconds later (nothing the card
//! was told before that is still running: Windows resets a card that takes
//! that long over one submission), and by then no list names it (the owners' rule, as for allocations of their own): a
//! texture's piece comes back when the last reference to the host texture
//! goes (an object of ours the texture holds says so), a staging buffer's
//! when the last list that recorded its copy is reset.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

/// Bytes of a block; pieces of more than half of one are not pooled (with
/// blocks made ahead, [`ahead`]: of more than one).
const BLOCK: u64 = 64 << 20;
/// Blocks with nothing in them that are kept.
const SPARE: usize = 2;
/// The same while the card has room ([`room`]): 4 GiB of them.
const SPARE_KEPT: usize = 64;
/// The same of blocks of staging buffers, which are the PC's memory: run
/// crash-1 held 5.2 GB of them.
const STAGING_KEPT: usize = 8;
/// How many are kept now.
static SPARES: AtomicUsize = AtomicUsize::new(SPARE);

/// Sets how many empty blocks stay, by the card's memory (budget and use):
/// [`SPARE_KEPT`] below [`crate::host::VRAM_EVICT_SOFT_PCT`]
/// ([`crate::host::keep_freed`]), `0` while raw physical VRAM is in
/// high-pressure hysteresis, else [`SPARE`]. Asked once a second.
pub(crate) fn room(memory: Option<[u64; 4]>) {
    let raw_high = crate::resource::raw_vram_pressure(memory).is_some_and(|(_, _, high)| high);
    let keep = crate::host::keep_freed();
    if !crate::residency::on() {
        let limit = spares_for(memory, keep, raw_high);
        SPARES.store(limit, Ordering::Relaxed);
        // Reclaim rested pieces and excess empty 64 MiB blocks on the 1-second
        // sweep too so drained heaps return to WDDM even when allocations pause
        // under VRAM pressure.
        if let Ok(mut pool) = TEXTURES.try_lock() {
            pool.reclaim(now(), limit, true);
        }
        return;
    }
    // (`VICTORY_RESIDENCY=1`, codec-d/cool_fox c64fc9b4/e94d277a: fewer
    // spares while the system is short of memory, and idle blocks let go now,
    // not at the pool's next allocation.)
    let pressure = crate::residency::pressured();
    let tex_limit = spares_for(memory, keep && !pressure, raw_high);
    let stage_base = spares_for(memory, keep && !pressure, false);
    SPARES.store(tex_limit, Ordering::Relaxed);
    let tick = now();
    if let Ok(mut pool) = TEXTURES.try_lock() {
        pool.reclaim(tick, tex_limit, true);
    }
    if let Ok(mut pool) = STAGING.try_lock() {
        let staging_limit = if crate::residency::critical_commit() {
            0
        } else if pressure {
            1
        } else {
            stage_base
        };
        pool.reclaim(tick, staging_limit, true);
    }
}

/// Empty-block retention cap: drops to `0` under raw physical VRAM hysteresis
/// so drained whole-empty 64 MiB heaps return to WDDM immediately and
/// `ripe.ahead` pre-allocation pauses.
fn spares_for(memory: Option<[u64; 4]>, keep: bool, raw_high: bool) -> usize {
    if raw_high {
        return 0;
    }
    match memory {
        Some([budget, used, ..])
            if keep
                && u128::from(used) * 100
                    < u128::from(budget) * u128::from(crate::host::VRAM_EVICT_SOFT_PCT) =>
        {
            SPARE_KEPT
        }
        _ => SPARE,
    }
}

#[cfg(test)]
fn spares(memory: Option<[u64; 4]>, keep: bool) -> usize {
    spares_for(memory, keep, false)
}

/// Milliseconds until a piece given back is used again. (It was three
/// frames, a tenth of a second while the title shows a loading screen at 30
/// a second, with up to 18 submissions in the air.)
const REST: u64 = 2500;

fn now() -> u64 {
    crate::host::now_ms()
}

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_POOL").is_none_or(|v| v != "0"))
}

/// How many blocks of textures are made ahead, so that one has been there
/// for a while when its first piece is taken (`VICTORY_RIPE_BLOCKS`, 12;
/// 0: none, and new blocks are used at once, as before).
///
/// A page fault of lists already on the card, in a texture made the same
/// tenth of a second as the first piece of a block made for it, lost the
/// device in runs n08 and n11 (2026-11-25): the title gives a slot that
/// those lists read by index a new texture. Why the card does not have the
/// new block's addresses yet is not known (n11 waited for `MakeResident`;
/// `indirect::tests::new_blocks_under_readers` never loses it). A block that
/// every submission begun before its making has left behind ([`Ripe`]) has
/// no such lists: pieces are cut from those first.
fn ahead() -> usize {
    static AHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AHEAD.get_or_init(|| std::env::var("VICTORY_RIPE_BLOCKS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(12))
}

/// Submissions begun and finished ([`crate::host::submissions`]), and how
/// many blocks to make ahead (0: every block is ripe at once).
#[derive(Clone, Copy)]
struct Ripe {
    started: u64,
    finished: u64,
    ahead: usize,
}

impl Ripe {
    const ALL: Ripe = Ripe { started: 0, finished: 0, ahead: 0 };

    fn now() -> Ripe {
        let (started, finished) = crate::host::submissions();
        // Not more than are kept ([`room`]): they would go and be made again.
        Ripe { started, finished, ahead: ahead().min(SPARES.load(Ordering::Relaxed)) }
    }

    fn is<B>(self, block: &Block<B>) -> bool {
        self.ahead == 0 || block.made <= self.finished
    }
}

/// Pieces cut from blocks that were not ripe, for want of a ripe one.
static UNRIPE: AtomicU64 = AtomicU64::new(0);

/// The free ranges of a block: start and length, never two that touch.
struct Ranges {
    free: BTreeMap<u64, u64>,
    /// Total free bytes in this block (equal to `free.values().sum()`).
    bytes: u64,
}

impl Ranges {
    fn new(size: u64) -> Ranges {
        Ranges {
            free: BTreeMap::from([(0, size)]),
            bytes: size,
        }
    }

    fn free_bytes(&self) -> u64 {
        self.bytes
    }

    /// Finds the smallest free span in this block that holds `len` bytes at a
    /// multiple of `align` (tie-breaking by lowest offset), returning
    /// `(range_start, range_size, aligned_offset)`.
    fn best_fit(&self, len: u64, align: u64) -> Option<(u64, u64, u64)> {
        if len == 0 {
            return None;
        }
        let align = align.max(1);
        let mut best: Option<(u64, u64, u64)> = None;
        for (&start, &size) in &self.free {
            let Some(at) = start.checked_next_multiple_of(align) else {
                continue;
            };
            let Some(end) = at.checked_add(len) else {
                continue;
            };
            if end <= start.saturating_add(size) {
                match best {
                    Some((best_start, best_size, _))
                        if (size, start) >= (best_size, best_start) => {}
                    _ => best = Some((start, size, at)),
                }
            }
        }
        best
    }

    /// Carves `[at, at + len)` out of the known free range `(start, size)`.
    fn take_at(&mut self, start: u64, size: u64, at: u64, len: u64) -> u64 {
        self.free.remove(&start);
        if at > start {
            self.free.insert(start, at - start);
        }
        if at + len < start + size {
            self.free.insert(at + len, start + size - at - len);
        }
        self.bytes = self.bytes.saturating_sub(len);
        at
    }

    /// Best-fit place for `len` bytes at a multiple of `align`.
    #[cfg(test)]
    fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (start, size, at) = self.best_fit(len, align)?;
        Some(self.take_at(start, size, at, len))
    }

    fn give(&mut self, mut start: u64, mut len: u64) {
        let added = len;
        if let Some((&before, &size)) = self.free.range(..start).next_back()
            && before + size == start
        {
            self.free.remove(&before);
            (start, len) = (before, len + size);
        }
        if let Some(size) = self.free.remove(&(start + len)) {
            len += size;
        }
        self.free.insert(start, len);
        self.bytes = self.bytes.saturating_add(added);
    }

    fn whole(&self, size: u64) -> bool {
        self.free.get(&0) == Some(&size)
    }
}

/// A part of a block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    block: usize,
    start: u64,
    len: u64,
}

struct Block<B> {
    memory: B,
    free: Ranges,
    /// The end of the highest piece ever taken: above it the memory is as
    /// it was made (zeros).
    used_to: u64,
    /// The newest submission begun when it was made ([`Ripe`]).
    made: u64,
    /// When that was (ms).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    since: u64,
}

struct Pool<B> {
    /// `None`: a block let go, its place to be used again.
    blocks: Vec<Option<Block<B>>>,
    /// Pieces given back, and when each may be used again.
    resting: VecDeque<(u64, Piece)>,
    /// The empty blocks it was last told to keep, and the most it keeps.
    spare: usize,
    most: usize,
    /// No new block is asked for before this time after one was refused
    /// ([`Pool::grow`]).
    retry_growth_at: u64,
}

impl<B: Clone> Pool<B> {
    const fn new(most: usize) -> Pool<B> {
        Pool { blocks: Vec::new(), resting: VecDeque::new(), spare: 0, most, retry_growth_at: 0 }
    }

    /// A piece of `len` bytes and the memory it lies in, and whether that
    /// part was ever used before; `make` makes a block when none has room.
    fn take(&mut self, len: u64, align: u64, now: u64, make: impl FnMut() -> Option<B>) -> Option<(Piece, B, bool)> {
        self.take_ripe(len, align, now, Ripe::ALL, make)
    }

    /// Picks the best block and free span for `(len, align)`:
    /// 1. Prefer higher-occupancy (smaller `free_bytes`) non-empty ripe blocks
    ///    and their smallest sufficient free span (best-fit), so low-occupancy
    ///    tail 64 MiB heaps stop taking new textures and drain to `whole(BLOCK)`
    ///    to return physical VRAM to WDDM.
    /// 2. When `non_empty_only` is `false`, fall back to whole-empty ripe blocks,
    ///    then (when `ripe.ahead > 0`) to the oldest unripe block.
    /// 3. When `non_empty_only` is `true`, never touch whole-empty blocks and
    ///    never allocate a new WDDM heap.
    fn pick_candidate(
        &self,
        len: u64,
        align: u64,
        ripe: Ripe,
        non_empty_only: bool,
    ) -> Option<(usize, u64, u64, u64)> {
        let mut best_ripe: Option<((bool, u64, u64, usize), (usize, u64, u64, u64))> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block.as_ref() else {
                continue;
            };
            if !ripe.is(block) {
                continue;
            }
            let is_empty = block.free.whole(BLOCK);
            if non_empty_only && is_empty {
                continue;
            }
            let Some((hole_start, hole_size, at)) = block.free.best_fit(len, align) else {
                continue;
            };
            let rank = (is_empty, block.free.free_bytes(), hole_size, i);
            if best_ripe.is_none_or(|(best_rank, _)| rank < best_rank) {
                best_ripe = Some((rank, (i, hole_start, hole_size, at)));
            }
        }
        if let Some((_, picked)) = best_ripe {
            return Some(picked);
        }
        if ripe.ahead > 0 {
            let mut best_unripe: Option<((u64, bool, u64, u64, usize), (usize, u64, u64, u64))> =
                None;
            for (i, block) in self.blocks.iter().enumerate() {
                let Some(block) = block.as_ref() else {
                    continue;
                };
                if ripe.is(block) {
                    continue;
                }
                let is_empty = block.free.whole(BLOCK);
                if non_empty_only && is_empty {
                    continue;
                }
                let Some((hole_start, hole_size, at)) = block.free.best_fit(len, align) else {
                    continue;
                };
                let rank = (block.made, is_empty, block.free.free_bytes(), hole_size, i);
                if best_unripe.is_none_or(|(best_rank, _)| rank < best_rank) {
                    best_unripe = Some((rank, (i, hole_start, hole_size, at)));
                }
            }
            if let Some((_, picked)) = best_unripe {
                return Some(picked);
            }
        }
        None
    }

    /// Returns whether any already-occupied block has a free span able to hold
    /// `(len, align)` without allocating a new block or touching whole-empty blocks.
    fn has_non_empty_hole(&self, len: u64, align: u64, ripe: Ripe) -> bool {
        len > 0 && len <= BLOCK && self.pick_candidate(len, align, ripe, true).is_some()
    }

    /// Sub-allocates only from existing non-empty blocks via best-fit clustering:
    /// never calls `make` to create a new WDDM heap and never consumes whole-empty
    /// spare blocks waiting to be returned to WDDM.
    fn take_existing_non_empty(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
    ) -> Option<(Piece, B, bool)> {
        if len == 0 || len > BLOCK {
            return None;
        }
        self.rested(now);
        let (i, hole_start, hole_size, at) = self.pick_candidate(len, align, ripe, true)?;
        let start = self.blocks[i]
            .as_mut()?
            .free
            .take_at(hole_start, hole_size, at, len);
        if ripe.ahead > 0 && !ripe.is(self.blocks[i].as_ref()?) {
            let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!("textures cut from blocks made while lists ran that still run: {n}");
            }
        }
        let block = self.blocks[i].as_mut()?;
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        Some((
            Piece {
                block: i,
                start,
                len,
            },
            block.memory.clone(),
            again,
        ))
    }

    /// The same from a block that is ripe, if one has room; `make` makes one
    /// ahead too, when few are empty.
    fn take_ripe(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        mut make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        // With blocks made ahead, what would be an allocation of its own (and
        // new when the card first reads it) lies in one of those too.
        if len == 0 || len > if ripe.ahead > 0 { BLOCK } else { BLOCK / 2 } {
            return None;
        }
        self.rested(now);
        let found = self.pick_candidate(len, align, ripe, false).and_then(
            |(i, hole_start, hole_size, at)| {
                let start = self.blocks[i]
                    .as_mut()?
                    .free
                    .take_at(hole_start, hole_size, at, len);
                Some((i, start))
            },
        );
        let (i, start) = match found {
            Some(found) => found,
            None => {
                let memory = self.grow(now, &mut make)?;
                let i = self.add(memory, ripe.started);
                let (hole_start, hole_size, at) =
                    self.blocks[i].as_ref()?.free.best_fit(len, align)?;
                let start = self.blocks[i]
                    .as_mut()?
                    .free
                    .take_at(hole_start, hole_size, at, len);
                (i, start)
            }
        };
        if ripe.ahead > 0 {
            if !ripe.is(self.blocks[i].as_ref()?) {
                let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!("textures cut from blocks made while lists ran that still run: {n}");
                }
            }
            let empty = self.blocks.iter().flatten().filter(|b| b.free.whole(BLOCK)).count();
            if empty < ripe.ahead
                && let Some(memory) = self.grow(now, &mut make)
            {
                self.add(memory, ripe.started);
            }
        }
        let block = self.blocks[i].as_mut()?;
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        Some((Piece { block: i, start, len }, block.memory.clone(), again))
    }

    /// A new block from `make`, unless one was refused less than 500 ms ago
    /// (codec-d/cool_fox 08e4574d: a refused 64 MiB block is not asked for
    /// again for each small texture; the blocks there are still used).
    fn grow(&mut self, now: u64, make: &mut impl FnMut() -> Option<B>) -> Option<B> {
        if now < self.retry_growth_at {
            return None;
        }
        let block = make();
        if block.is_none() {
            self.retry_growth_at = now.saturating_add(500);
        }
        block
    }

    fn add(&mut self, memory: B, made: u64) -> usize {
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.blocks.push(None);
                self.blocks.len() - 1
            }
        };
        self.blocks[i] = Some(Block { memory, free: Ranges::new(BLOCK), used_to: 0, made, since: crate::host::now_ms() });
        i
    }

    fn give(&mut self, piece: Piece, now: u64) {
        self.resting.push_back((now + REST, piece));
    }

    /// Makes the pieces that rested long enough free, and lets go of empty
    /// blocks past [`SPARE`].
    fn rested(&mut self, now: u64) {
        self.reclaim(now, SPARES.load(Ordering::Relaxed), false);
    }

    /// [`Pool::rested`] with at most `limit` empty blocks kept; `sweep`: the
    /// empty blocks are looked at even if no piece came back.
    fn reclaim(&mut self, now: u64, limit: usize, sweep: bool) {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(block) = self.blocks[piece.block].as_mut() {
                block.free.give(piece.start, piece.len);
                any |= block.free.whole(BLOCK);
            }
        }
        // (Without a piece given back too, once there is less room.)
        let most = limit.min(self.most);
        if sweep || any || most < self.spare {
            self.spare = most;
            let mut spare = 0;
            for block in &mut self.blocks {
                if block.as_ref().is_some_and(|b| b.free.whole(BLOCK)) {
                    spare += 1;
                    if spare > most {
                        *block = None;
                    }
                }
            }
        }
    }

    fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }

    /// Splits immediately free pool bytes into `(whole_spare_bytes, fragmented_free_bytes)`:
    /// completely empty 64 MiB blocks returnable to WDDM vs. free holes scattered
    /// inside partially occupied blocks (excluding pieces still in `resting`).
    fn spare_and_fragmented_room(&self) -> (u64, u64) {
        let mut whole_spare = 0u64;
        let mut fragmented = 0u64;
        for block in self.blocks.iter().flatten() {
            if block.free.whole(BLOCK) {
                whole_spare += BLOCK;
            } else {
                fragmented += block.free.free_bytes();
            }
        }
        (whole_spare, fragmented)
    }
}

/// A mapped upload buffer's memory.
#[derive(Clone)]
struct Upload(ID3D12Resource, *mut u8);
// SAFETY: the mapping is written by the owners of its pieces, each its own.
unsafe impl Send for Upload {}

static TEXTURES: Mutex<Pool<ID3D12Heap>> = Mutex::new(Pool::new(SPARE_KEPT));
static STAGING: Mutex<Pool<Upload>> = Mutex::new(Pool::new(STAGING_KEPT));

fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes of the pieces textures hold now.
static HELD: AtomicU64 = AtomicU64::new(0);

/// Bytes of the texture blocks no texture holds: the card's memory in use
/// by us that new textures take first.
pub(crate) fn texture_room() -> u64 {
    (lock(&TEXTURES).blocks() as u64 * BLOCK).saturating_sub(HELD.load(Ordering::Relaxed))
}

/// Splits free texture-pool memory into `(whole_spare_bytes, fragmented_free_bytes)`:
/// whole-empty 64 MiB blocks returnable to WDDM vs. free holes inside partially
/// occupied blocks.
pub(crate) fn spare_and_fragmented_room() -> (u64, u64) {
    lock(&TEXTURES).spare_and_fragmented_room()
}

/// Returns whether an already-occupied texture-pool block has a reusable free
/// hole for `(size, alignment)` without creating a new WDDM heap or touching
/// whole-empty spare blocks.
pub(crate) fn has_reusable_hole(size: u64, alignment: u64) -> bool {
    if !on() {
        return false;
    }
    let ripe = Ripe::now();
    let mut pool = lock(&TEXTURES);
    pool.rested(now());
    pool.has_non_empty_hole(size, alignment, ripe)
}

/// For the memory line: blocks of textures and of staging buffers, in MiB.
pub(crate) fn memory() -> [u64; 2] {
    [lock(&TEXTURES).blocks() as u64 * (BLOCK >> 20), lock(&STAGING).blocks() as u64 * (BLOCK >> 20)]
}

/// Part of a block of mapped upload memory, back in the pool when dropped.
pub(crate) struct StagingPiece(Piece);

impl Drop for StagingPiece {
    fn drop(&mut self) {
        lock(&STAGING).give(self.0, now());
    }
}

/// `bytes` of upload memory, all zeros: the buffer it lies in, where in
/// that buffer, and its mapping. `None`: not pooled.
pub(crate) fn staging(host: &crate::host::Host, bytes: u64) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
    if !on() {
        return None;
    }
    // Copies from a buffer begin at multiples of 512.
    let bytes = bytes.next_multiple_of(512);
    let (piece, memory, again) = lock(&STAGING).take(bytes, 512, now(), || {
        // (`VICTORY_RESIDENCY=1`: no new block for a texture picked by index
        // while commit is short; those a draw samples now still get one.)
        let speculative = crate::texture::PICKING.get().is_some() && !crate::texture::BAKING.get();
        if speculative && !crate::residency::allow_staging_growth(false) {
            return None;
        }
        let (buffer, data) = host.upload_buffer(BLOCK)?;
        crate::host::name(&buffer, || "staging of textures (a block)".into());
        Some(Upload(buffer, data))
    })?;
    let owned = StagingPiece(piece);
    // SAFETY: the piece lies in the block's mapping.
    let data = unsafe { memory.1.add(piece.start as usize) };
    if again {
        // What is not detiled into it (an unmapped texture) uploads zeros,
        // as from a buffer of its own.
        // SAFETY: the piece is ours alone.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
    }
    Some((owned, memory.0, piece.start, data))
}

/// An object the host texture holds and releases when it goes: its piece
/// is free then.
#[repr(C)]
struct Reclaim {
    methods: &'static [*const (); 3],
    references: AtomicU32,
    piece: Piece,
}

struct Methods([*const (); 3]);
// SAFETY: function addresses only.
unsafe impl Sync for Methods {}

static RECLAIM: Methods = Methods([query as *const (), add_ref as *const (), release as *const ()]);

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: COM's arguments.
    unsafe {
        if out.is_null() {
            return HRESULT(0x8000_4003_u32 as i32);
        }
        if iid.as_ref().is_some_and(|iid| *iid == IUnknown::IID) {
            add_ref(this);
            *out = this;
            return HRESULT(0);
        }
        *out = std::ptr::null_mut();
    }
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: one of ours, alive while a reference to it is.
    unsafe { (*this.cast::<Reclaim>()).references.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: as above; the last reference frees the box it was made as.
    unsafe {
        let left = (*this.cast::<Reclaim>()).references.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            let gone = Box::from_raw(this.cast::<Reclaim>());
            HELD.fetch_sub(gone.piece.len, Ordering::Relaxed);
            lock(&TEXTURES).give(gone.piece, now());
        }
        left
    }
}

/// Whether textures of 64 KiB and less are placed at multiples of 4 KiB in
/// their block, where the host allows it (`VICTORY_POOL_SMALL=1`; otherwise
/// at multiples of 64 KiB, as every other texture).
///
/// The two device losses of the night of 2026-11-25 with the pool in the
/// report (runs steps-ab2 and start-3) faulted in the same kind of texture:
/// 256x256, block-compressed, 7 levels, placed at a 4 KiB multiple that is no
/// 64 KiB one (offsets 0x406000 and 0x3fe6000 of their blocks), never used
/// before, read by the card the moment its slot was made; the descriptor's
/// base of both lies 0xa000 into the piece, on a 64 KiB multiple, and its
/// first level of 32 KiB from there ends 0x2000 past the piece. Why the card
/// faulted there is not proven; textures at 64 KiB multiples have none of
/// that, for some memory (at most 60 KiB a small texture).
fn small_placement() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_POOL_SMALL") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_SMALL").is_some_and(|v| v != "0"))
}

/// Diagnostics: whether the card reads a pooled texture inside the piece it
/// was placed in. A block's address is taken as the lowest of its textures'
/// descriptor bases less their pieces' starts (a texture's first level may
/// lie past its piece's start, never before it); a texture whose first level
/// the card would read from past its piece's end is said loudly.
#[cfg(feature = "diagnostics")]
fn where_the_card_reads(heap: &ID3D12Heap, texture: &ID3D12Resource, piece: Piece) {
    static BLOCKS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    static SAID: AtomicU64 = AtomicU64::new(0);
    let Some(base) = crate::book::address(texture) else { return };
    let Some(seen) = base.checked_sub(piece.start) else { return };
    let raw = heap.as_raw() as usize;
    let block = {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        match blocks.iter_mut().find(|b| b.0 == raw) {
            Some(b) => {
                b.1 = b.1.min(seen);
                b.1
            }
            None => {
                if blocks.len() >= 4096 {
                    blocks.remove(0);
                }
                blocks.push((raw, seen));
                seen
            }
        }
    };
    let into = base - block - piece.start;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    // The first level's bytes, as a copy would have them.
    let mut level = 0u64;
    if let Some(host) = crate::host::made_host() {
        // SAFETY: plain query of a valid description.
        unsafe { host.device.GetCopyableFootprints(&desc, 0, 1, 0, None, None, None, Some(&raw mut level)) };
    }
    if into + level > piece.len {
        let n = SAID.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 50 || n.is_power_of_two() {
            // (Debug only: it assumes a texture's address is its block's
            // start and its piece's offset, which this driver does not
            // keep: 56 false alarms in run pool-1.)
            tracing::debug!(
                times = n,
                base = format_args!("{base:#x}"),
                piece_start = format_args!("{:#x}", piece.start),
                piece_len = format_args!("{:#x}", piece.len),
                into = format_args!("{into:#x}"),
                first_level_bytes = format_args!("{level:#x}"),
                width = desc.Width,
                height = desc.Height,
                levels = desc.MipLevels,
                format = desc.Format.0,
                "POOL: the card reads a pooled texture's first level past the end of the piece it was placed in"
            );
        }
    }
    // Every placement offset and where the first level lies, a few of them.
    static SHOWN: AtomicU64 = AtomicU64::new(0);
    if into > 0 && SHOWN.fetch_add(1, Ordering::Relaxed) < 40 {
        tracing::info!(
            into = format_args!("{into:#x}"),
            piece_start = format_args!("{:#x}", piece.start),
            piece_len = format_args!("{:#x}", piece.len),
            width = desc.Width,
            height = desc.Height,
            levels = desc.MipLevels,
            format = desc.Format.0,
            alignment = desc.Alignment,
            "pool: a texture's first level lies past its piece's start"
        );
    }
}

/// Textures made in the last second ([`fresh`]), and when.
static RECENT: Mutex<VecDeque<(usize, u64)>> = Mutex::new(VecDeque::new());

/// Whether `texture` was made in a block in the last second.
pub(crate) fn fresh(texture: &ID3D12Resource) -> bool {
    let raw = texture.as_raw() as usize;
    let since = now().saturating_sub(1000);
    RECENT.lock().unwrap_or_else(|e| e.into_inner()).iter().rev().take_while(|r| r.1 >= since).any(|r| r.0 == raw)
}

fn made_now(texture: &ID3D12Resource) {
    let now = now();
    let raw = texture.as_raw() as usize;
    {
        let mut recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
        while recent.front().is_some_and(|r| r.1 + 1000 < now) {
            recent.pop_front();
        }
        recent.push_back((raw, now));
    }
    let begun = crate::host::submissions().0;
    NEWEST_BEGUN.fetch_max(begun, Ordering::Relaxed);
    let mut unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    let finished = crate::host::submissions().1;
    // Those every submission begun before them has left behind are no
    // one's concern any more (and a raw pointer may come again).
    unseen.retain(|_, b| *b > finished);
    unseen.insert(raw, begun);
}

/// Pooled textures made while submissions begun before them may still run:
/// the submissions begun when each was made.
static UNSEEN: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, u64>>> = std::sync::LazyLock::new(Default::default);
/// The newest of those numbers (nothing to look up once it has finished).
static NEWEST_BEGUN: AtomicU64 = AtomicU64::new(0);

/// Whether a texture was made in a block while a submission begun before
/// it may still run ([`begun_before`]).
pub(crate) fn any_unseen() -> bool {
    NEWEST_BEGUN.load(Ordering::Relaxed) > crate::host::submissions().1
}

/// The submissions begun when `texture` was made in a block, if one of
/// those may still be on the card: lists already sent then must not find it
/// in a slot they read by index ([`crate::descriptor::host::show_new_later`]).
/// Every device loss with the pool in the report (runs n08, n11, n15,
/// steps-ab2, start-3 and the user's play-1125-2107) was a page fault at the
/// first byte of a texture made that very millisecond, its title slot
/// changed "at once" while lists sent before it read that slot by index.
pub(crate) fn begun_before(texture: &ID3D12Resource) -> Option<u64> {
    if !any_unseen() {
        return None;
    }
    let finished = crate::host::submissions().1;
    let raw = texture.as_raw() as usize;
    let unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    unseen.get(&raw).copied().filter(|&begun| begun > finished)
}

/// Makes `texture` give `piece` back when the last reference to it goes.
fn give_back_with(texture: &ID3D12Resource, piece: Piece) {
    const KEY: GUID = GUID::from_u128(0x6f0e2f0c_5a59_4b1e_9a3f_7f4d2c9be713);
    let made = Box::into_raw(Box::new(Reclaim { methods: &RECLAIM.0, references: AtomicU32::new(1), piece }));
    // SAFETY: a COM object of ours with one reference, which this takes and
    // drops at the end; the texture takes its own.
    unsafe {
        let object = IUnknown::from_raw(made.cast());
        if texture.SetPrivateDataInterface(&KEY, &object).is_err() {
            // Never given back: the piece stays taken, nothing is reused
            // under a texture that may be alive.
            std::mem::forget(object);
        }
    }
}

/// A texture of `desc` (no target, depth or unordered access) in a block;
/// `None`: not pooled, or the host refused.
///
/// # Safety
/// `desc` is a valid description.
pub(crate) unsafe fn texture(
    device: &ID3D12Device,
    desc: *const D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    if !on() {
        return None;
    }
    // Small textures lie at multiples of 4 KiB where the host allows it
    // (the top mip is 64 KiB at most) and [`small_placement`] says so: 64 KiB
    // each otherwise.
    // SAFETY: the caller's description.
    let mut d = unsafe { *desc };
    d.Alignment = if small_placement() { u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) } else { 0 };
    // SAFETY: plain queries.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    if info.SizeInBytes == u64::MAX || info.Alignment != d.Alignment {
        d.Alignment = 0;
        info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    }
    if info.SizeInBytes == u64::MAX {
        return None;
    }
    let align = info.Alignment.max(1);
    let now_ms = now();
    let ripe = Ripe::now();
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (piece, heap, again) = if crate::resource::allow_new_texture_heap_lately() {
        lock(&TEXTURES).take_ripe(info.SizeInBytes, align, now_ms, ripe, || {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: BLOCK,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | crate::host::not_resident(),
            };
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: plain creation.
            if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
                crate::residency::allocation_error(&e);
                tracing::warn!(error = %e, "no block of texture memory");
            }
            if let Some(h) = &heap
                && let Err(e) =
                    crate::host::resident(device, h).and_then(|()| crate::host::mapped(device))
            {
                tracing::warn!(error = %e, "a block of texture memory is not resident");
                return None;
            }
            heap
        })?
    } else {
        lock(&TEXTURES).take_existing_non_empty(info.SizeInBytes, align, now_ms, ripe)?
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: the piece is no one else's; the heap lives while a piece of
    // it is taken.
    let result = unsafe { device.CreatePlacedResource(&heap, piece.start, &d, state, None, &mut made) };
    match (result, made) {
        (Ok(()), Some(made)) => {
            HELD.fetch_add(piece.len, Ordering::Relaxed);
            give_back_with(&made, piece);
            made_now(&made);
            #[cfg(feature = "diagnostics")]
            {
                crate::book::made(&made, || "a texture in a block".into());
                where_the_card_reads(&heap, &made, piece);
                crate::book::block(&heap, &made, piece.start, BLOCK);
                crate::book::note(made.as_raw() as usize, || {
                    let block = lock(&TEXTURES).blocks.get(piece.block).and_then(Option::as_ref).map(|b| (b.made, b.since));
                    let (submission, since) = block.unwrap_or_default();
                    format!(
                        "; piece {:#x}+{:#x} of block {} {:#x} made {:.1} s before it at submission {submission}, used before: {again}, made at submissions {:?}",
                        piece.start,
                        piece.len,
                        piece.block,
                        heap.as_raw() as usize,
                        now().saturating_sub(since) as f64 / 1e3,
                        crate::host::submissions()
                    )
                });
            }
            Some(made)
        }
        _ => {
            lock(&TEXTURES).give(piece, now());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_block_is_not_retried_for_each_small_upload() {
        let mut pool: Pool<u32> = Pool::new(2);
        let calls = std::cell::Cell::new(0);
        let refuse = || {
            calls.set(calls.get() + 1);
            None
        };
        for size in [704008, 179720, 179720, 24080, 48648, 524288, 89616, 351760] {
            assert!(pool.take(size, 512, 12, refuse).is_none());
        }
        assert_eq!(calls.get(), 1);
        assert!(pool.take(24080, 512, 1012, || Some(1)).is_some());
    }

    #[test]
    fn budget_tick_reclaims_idle_blocks_without_reusing_live_or_resting_pieces() {
        let mut pool = Pool::new(STAGING_KEPT);
        let (held, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        let (resting, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(2)).unwrap();
        pool.give(resting, 0);
        for i in 0..12 {
            pool.add(i, 0);
        }
        pool.reclaim(0, SPARE_KEPT, true);
        assert_eq!(pool.blocks(), STAGING_KEPT + 1);
        pool.reclaim(REST - 1, SPARE, true);
        assert_eq!(pool.blocks(), SPARE + 1);
        assert_eq!(pool.resting.len(), 1);
        pool.reclaim(REST, 0, true);
        assert_eq!(pool.blocks(), 1);
        assert!(pool.resting.is_empty());
        pool.give(held, REST);
        pool.reclaim(2 * REST, 0, true);
        assert_eq!(pool.blocks(), 0);
    }

    #[test]
    fn ranges_are_cut_aligned_and_join_again() {
        let mut r = Ranges::new(1024);
        assert_eq!(r.take(100, 64), Some(0));
        assert_eq!(r.take(100, 64), Some(128));
        assert_eq!(r.take(28, 4), Some(100));
        assert_eq!(r.take(2000, 4), None);
        r.give(128, 100);
        r.give(0, 100);
        assert!(!r.whole(1024));
        r.give(100, 28);
        assert!(r.whole(1024));
    }

    #[test]
    fn empty_blocks_stay_while_the_card_has_room() {
        assert_eq!(spares(Some([100, 84, 0, 0]), true), SPARE_KEPT);
        assert_eq!(spares(Some([100, 85, 0, 0]), true), SPARE);
        assert_eq!(spares(Some([100, 10, 0, 0]), false), SPARE);
        assert_eq!(spares(None, true), SPARE);
    }

    #[test]
    fn pieces_never_overlap_and_rest_before_they_are_used_again() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let mut taken: Vec<Piece> = Vec::new();
        for i in 0..500u64 {
            let (p, _, again) = pool.take(65536 * (1 + i % 7), 65536, 0, make).unwrap();
            assert!(!again);
            assert!(taken.iter().all(|q| q.block != p.block || p.start + p.len <= q.start || q.start + q.len <= p.start));
            taken.push(p);
        }
        let blocks = pool.blocks();
        assert!(blocks >= 2);
        // Given back at 10: not used again before 10 + REST.
        let first = taken[0];
        pool.give(first, 10);
        let (p, _, _) = pool.take(first.len, 65536, 10 + REST - 1, make).unwrap();
        assert_ne!((p.block, p.start), (first.block, first.start));
        let (q, _, again) = pool.take(first.len, 65536, 10 + REST, make).unwrap();
        assert_eq!((q.block, q.start, again), (first.block, first.start, true));
        // Everything back: the empty blocks past the spare ones go.
        for p in taken.drain(1..).chain([p, q]) {
            pool.give(p, 20);
        }
        pool.rested(20 + REST);
        assert_eq!(pool.blocks(), blocks.min(SPARE));
        // Too big for a block, and a block that cannot be made.
        assert!(pool.take(BLOCK, 65536, 30, make).is_none());
        let mut none: Pool<u32> = Pool::new(SPARE);
        assert!(none.take(65536, 65536, 0, || None).is_none());
    }

    #[test]
    fn pieces_come_from_blocks_older_than_the_lists_that_run() {
        let mut pool: Pool<u64> = Pool::new(SPARE_KEPT);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let at = |started, finished| Ripe { started, finished, ahead: 2 };
        let take = |pool: &mut Pool<u64>, ripe| pool.take_ripe(BLOCK / 2, 65536, 0, ripe, make).unwrap().1;
        // Nothing runs: at once, and two are made ahead, one a time.
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 2);
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 3);
        // 4 is made while submission 5 runs.
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(made.get(), 4);
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        // None is ripe: the oldest; it is ripe once 5 has finished.
        assert_eq!(take(&mut pool, at(7, 4)), 4);
        assert_eq!(take(&mut pool, at(7, 5)), 4);
        // A whole block's worth too.
        assert_eq!(pool.take_ripe(BLOCK, 65536, 0, at(8, 7), make).unwrap().1, 5);
        // Off: the first with room, whatever its age, none ahead.
        let n = made.get();
        assert_eq!(pool.take(BLOCK / 2, 65536, 0, make).unwrap().1, 6);
        assert_eq!(made.get(), n);
        assert!(pool.take(BLOCK, 65536, 0, make).is_none());
    }

    /// A texture placed in a heap holds no reference to it that its count
    /// shows (this host, 2026-11-25: 1 before placing, 1 with the texture):
    /// the pool must not count on one, and does not (a block goes only with
    /// every piece back and rested, [`Pool::rested`]). Said here, and the
    /// count is the same again once the texture went.
    #[test]
    fn a_placed_texture_holds_its_heap() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: 1 << 20,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        unsafe { host.device.CreateHeap(&desc, &mut heap) }.unwrap();
        let heap = heap.unwrap();
        let before = crate::host::references(&heap);
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a live heap, a valid description.
        unsafe { host.device.CreatePlacedResource(&heap, 0, &texture_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made) }.unwrap();
        let made = made.unwrap();
        let held = crate::host::references(&heap);
        println!("heap references: {before} before placing, {held} with a texture in it");
        assert!(held >= before);
        drop(made);
        assert_eq!(crate::host::references(&heap), before);
    }

    /// A pooled texture's piece comes back when its last reference goes, not
    /// before; its staging piece begins as zeros.
    #[test]
    fn a_texture_gives_its_piece_back_when_it_goes() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: a valid description.
        let Some(t) = (unsafe { texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON) }) else { return };
        assert_eq!(crate::host::references(&t), 1, "what the texture holds is no reference to it");
        let resting = || lock(&TEXTURES).resting.len();
        let before = resting();
        let other = t.clone();
        drop(t);
        assert_eq!(resting(), before);
        drop(other);
        assert_eq!(resting(), before + 1);
        // SAFETY: the piece's mapping, 4096 bytes of it ours.
        unsafe {
            let (piece, _, at, data) = staging(host, 4096).unwrap();
            assert!(at.is_multiple_of(512));
            std::ptr::write_bytes(data, 0xAB, 4096);
            drop(piece);
            lock(&STAGING).rested(u64::MAX);
            let (_piece, _, _, data) = staging(host, 4096).unwrap();
            assert!((0..4096).all(|i| *data.add(i) == 0));
            // SAFETY: plain query.
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }

    #[test]
    fn best_fit_drains_low_occupancy_tail_block_to_whole_free() {
        const MIB: u64 = 1 << 20;
        let mut pool: Pool<u32> = Pool::new(SPARE_KEPT);
        let mut next = 0u32;
        let mut make = || {
            next += 1;
            Some(next)
        };
        // Fill block 1 completely with 4 x 16 MiB pieces (64 MiB).
        let mut b0 = Vec::new();
        for _ in 0..4 {
            let (p, id, _) = pool.take(16 * MIB, 65536, 0, &mut make).unwrap();
            assert_eq!(id, 1);
            b0.push(p);
        }
        // Tail block 2 has low occupancy: a single 4 MiB piece (60 MiB free).
        let (tail_piece, tail_id, _) = pool.take(4 * MIB, 65536, 0, &mut make).unwrap();
        assert_eq!(tail_id, 2);
        assert_eq!(pool.blocks(), 2);

        // Free one 16 MiB piece in block 1. While resting, only block 2's 60 MiB
        // is immediately free; once REST elapses, fragmented room grows to 76 MiB.
        pool.give(b0.remove(1), 10);
        assert_eq!(pool.spare_and_fragmented_room(), (0, 60 * MIB));
        pool.reclaim(10 + REST, SPARE_KEPT, false);
        assert_eq!(pool.spare_and_fragmented_room(), (0, 76 * MIB));

        // Subsequent 4 MiB and 8 MiB placements must pack into higher-occupancy
        // block 1 rather than refilling low-occupancy tail block 2.
        let (c1, id1, _) = pool.take(4 * MIB, 65536, 10 + REST, &mut make).unwrap();
        let (c2, id2, _) = pool.take(8 * MIB, 65536, 10 + REST, &mut make).unwrap();
        assert_eq!(
            id1, 1,
            "higher-occupancy block 1 must absorb new placements first"
        );
        assert_eq!(
            id2, 1,
            "higher-occupancy block 1 must continue clustering placements"
        );
        assert!(c1.start.is_multiple_of(65536) && c2.start.is_multiple_of(65536));

        // Releasing tail block 2's lone piece drains it to whole(BLOCK); with
        // spare limit 0 under VRAM pressure it is reclaimed immediately.
        pool.give(tail_piece, 20);
        pool.reclaim(20 + REST, 0, true);
        assert_eq!(
            pool.blocks(),
            1,
            "drained low-occupancy tail block should return to WDDM"
        );
        assert_eq!(pool.spare_and_fragmented_room(), (0, 4 * MIB));
    }

    #[test]
    fn existing_non_empty_only_reuses_holes_without_making_or_touching_empty_blocks() {
        const MIB: u64 = 1 << 20;
        let mut pool: Pool<u32> = Pool::new(SPARE_KEPT);
        let mut made = 0u32;
        let mut make = || {
            made += 1;
            Some(made)
        };
        // Block 1 holds 62 MiB (2 MiB fragmented hole); block 2 becomes whole-empty spare.
        let (p_big, id_a, _) = pool.take(32 * MIB, 65536, 0, &mut make).unwrap();
        let (p_mid, id_b, _) = pool.take(30 * MIB, 65536, 0, &mut make).unwrap();
        assert_eq!((id_a, id_b), (1, 1));
        let (p_tmp, id_empty, _) = pool.take(16 * MIB, 65536, 0, &mut make).unwrap();
        assert_eq!(id_empty, 2);
        pool.give(p_tmp, 10);
        pool.reclaim(10 + REST, SPARE_KEPT, false);
        assert_eq!(pool.spare_and_fragmented_room(), (BLOCK, 2 * MIB));

        // 1 MiB fits block 1's hole; 8 MiB must NOT touch whole-empty block 2.
        assert!(pool.has_non_empty_hole(MIB, 65536, Ripe::ALL));
        assert!(!pool.has_non_empty_hole(8 * MIB, 65536, Ripe::ALL));
        assert!(
            pool.take_existing_non_empty(8 * MIB, 65536, 10 + REST, Ripe::ALL)
                .is_none()
        );

        let before_made = made;
        let (reused, reused_id, _) = pool
            .take_existing_non_empty(MIB, 65536, 10 + REST, Ripe::ALL)
            .unwrap();
        assert_eq!(reused_id, 1);
        assert_eq!(made, before_made);
        assert_eq!(reused.len, MIB);
        assert_eq!(pool.spare_and_fragmented_room(), (BLOCK, MIB));
        let _ = (p_big, p_mid);
    }

    #[test]
    fn spares_drops_to_zero_under_raw_vram_hysteresis() {
        const MIB: u64 = 1 << 20;
        let budget = 13_579 * MIB;
        let used_86 = budget * 86 / 100;
        let used_84 = budget * 84 / 100;
        let used_813 = budget * 813 / 1000;

        // Crossing 86% enters raw high pressure -> spare retention drops to 0.
        let high_enter = crate::resource::raw_vram_high_for(budget, used_86, false);
        assert!(high_enter);
        assert_eq!(
            spares_for(Some([budget, used_86, 0, 0]), true, high_enter),
            0
        );

        // Inside the 82%..86% hysteresis band -> stays at 0 without oscillating.
        let high_hold = crate::resource::raw_vram_high_for(budget, used_84, high_enter);
        assert!(high_hold);
        assert_eq!(
            spares_for(Some([budget, used_84, 0, 0]), true, high_hold),
            0
        );

        // Below 82% exit line with margin <= 85% -> exits pressure and restores spares.
        let high_exit = crate::resource::raw_vram_high_for(budget, used_813, high_hold);
        assert!(!high_exit);
        assert_eq!(
            spares_for(Some([budget, used_813, 0, 0]), true, high_exit),
            SPARE_KEPT
        );
    }
}
