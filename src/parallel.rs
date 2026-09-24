//! The thread pool used for parallel work.
//!
//! medrs runs its parallel loops on its own rayon pool rather than rayon's
//! global pool, so that it keeps working in processes created with `fork()`,
//! such as PyTorch `DataLoader` workers. A forked child inherits the global
//! pool's bookkeeping but not its threads, so any later parallel call in the
//! child would wait forever. medrs recreates its pool whenever the process id
//! changes.
//!
//! When medrs is called from inside a rayon pool (for example your own
//! `ThreadPool::install`), it runs on that pool instead.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

struct Entry {
    pid: u32,
    threads: usize,
    pool: rayon::ThreadPool,
}

/// The current pool. Entries are leaked, never freed: a pool that is replaced
/// may still be in use by another thread, and after `fork()` its threads do
/// not exist, so it could not be shut down anyway.
static CURRENT: AtomicPtr<Entry> = AtomicPtr::new(std::ptr::null_mut());
static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the number of threads medrs uses for parallel work (0 restores the
/// default, the number of available CPUs).
///
/// Takes effect for the next parallel operation. Each change starts a new
/// pool and the previous pool's threads stay idle until the process exits,
/// so set this once at startup.
pub fn set_num_threads(threads: usize) {
    REQUESTED_THREADS.store(threads, Ordering::Relaxed);
}

/// Number of threads medrs uses for parallel work.
pub fn num_threads() -> usize {
    match REQUESTED_THREADS.load(Ordering::Relaxed) {
        0 => std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        n => n,
    }
}

#[allow(unsafe_code)]
fn pool() -> Option<&'static rayon::ThreadPool> {
    let pid = std::process::id();
    let threads = num_threads();
    let current = CURRENT.load(Ordering::Acquire);
    // SAFETY: non-null pointers in `CURRENT` come from `Box::leak` below and
    // are never freed, so they are valid for the rest of the process.
    if let Some(entry) = unsafe { current.as_ref() } {
        if entry.pid == pid && entry.threads == threads {
            return Some(&entry.pool);
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("medrs-{i}"))
        .build()
        .ok()?;
    let entry: &'static Entry = Box::leak(Box::new(Entry { pid, threads, pool }));
    let new = std::ptr::from_ref(entry).cast_mut();
    match CURRENT.compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Some(&entry.pool),
        // Another thread installed a pool first; use it (ours stays leaked
        // but idle, which only happens under a race at startup).
        // SAFETY: as above.
        Err(winner) => unsafe { winner.as_ref() }.map(|e| &e.pool),
    }
}

/// Run `op` on the medrs pool (or the current rayon pool, if called from one).
pub(crate) fn install<R: Send>(op: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return op();
    }
    match pool() {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn runs_parallel_work_and_nests() {
        let total: u64 = install(|| (0..10_000u64).into_par_iter().sum());
        assert_eq!(total, 49_995_000);
        let nested: u64 = install(|| install(|| (0..10u64).into_par_iter().sum()));
        assert_eq!(nested, 45);
        assert!(num_threads() >= 1);
    }
}
