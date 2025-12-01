//! Memory pool for buffer reuse in transform pipelines.
//!
//! This module provides a thread-local memory pool that allows transforms
//! to reuse allocated buffers, reducing allocation overhead in pipelines.

use std::cell::RefCell;
use std::collections::HashMap;

/// A thread-local memory pool for f32 buffers.
///
/// Buffers are keyed by their capacity (rounded up to a power of 2),
/// allowing efficient reuse across operations of similar sizes.
pub struct MemoryPool {
    /// Pool of available buffers, keyed by capacity.
    buffers: HashMap<usize, Vec<Vec<f32>>>,
    /// Maximum number of buffers to keep per size class.
    max_per_class: usize,
    /// Total memory budget (in bytes).
    budget: usize,
    /// Current memory usage (in bytes).
    current_usage: usize,
}

impl MemoryPool {
    /// Create a new memory pool with the given budget (in bytes).
    pub fn new(budget: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            max_per_class: 4,
            budget,
            current_usage: 0,
        }
    }

    /// Create a memory pool with default 256MB budget.
    pub fn default_budget() -> Self {
        Self::new(256 * 1024 * 1024)
    }

    /// Round up to the next power of 2 for binning.
    fn round_up_pow2(n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        let mut v = n - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        v + 1
    }

    /// Acquire a buffer of at least the given length.
    ///
    /// Returns a buffer from the pool if available, or allocates a new one.
    /// The buffer is zeroed before returning.
    pub fn acquire(&mut self, len: usize) -> Vec<f32> {
        let capacity = Self::round_up_pow2(len);

        // Try to get from pool
        if let Some(buffers) = self.buffers.get_mut(&capacity) {
            if let Some(mut buf) = buffers.pop() {
                self.current_usage -= buf.capacity() * std::mem::size_of::<f32>();
                buf.clear();
                buf.resize(len, 0.0);
                return buf;
            }
        }

        // Allocate new
        vec![0.0f32; len]
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// The buffer may be dropped if the pool is full or over budget.
    pub fn release(&mut self, buf: Vec<f32>) {
        let capacity = buf.capacity();
        let size = capacity * std::mem::size_of::<f32>();

        // Check if we're over budget
        if self.current_usage + size > self.budget {
            // Drop the buffer
            return;
        }

        let bin = Self::round_up_pow2(capacity);
        let buffers = self.buffers.entry(bin).or_insert_with(Vec::new);

        // Check max per class
        if buffers.len() >= self.max_per_class {
            return;
        }

        self.current_usage += size;
        buffers.push(buf);
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.current_usage = 0;
    }

    /// Get the current memory usage in bytes.
    pub fn usage(&self) -> usize {
        self.current_usage
    }

    /// Get the number of pooled buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.values().map(|v| v.len()).sum()
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::default_budget()
    }
}

// Thread-local pool for zero-allocation transforms
thread_local! {
    static POOL: RefCell<MemoryPool> = RefCell::new(MemoryPool::default_budget());
}

/// Acquire a buffer from the thread-local pool.
pub fn acquire_buffer(len: usize) -> Vec<f32> {
    POOL.with(|pool| pool.borrow_mut().acquire(len))
}

/// Return a buffer to the thread-local pool.
pub fn release_buffer(buf: Vec<f32>) {
    POOL.with(|pool| pool.borrow_mut().release(buf));
}

/// Clear the thread-local pool.
pub fn clear_pool() {
    POOL.with(|pool| pool.borrow_mut().clear());
}

/// Get the current thread-local pool usage in bytes.
pub fn pool_usage() -> usize {
    POOL.with(|pool| pool.borrow().usage())
}

/// A guard that returns a buffer to the pool when dropped.
pub struct PooledBuffer {
    buffer: Option<Vec<f32>>,
}

impl PooledBuffer {
    /// Create a new pooled buffer of the given length.
    pub fn new(len: usize) -> Self {
        Self {
            buffer: Some(acquire_buffer(len)),
        }
    }

    /// Get a reference to the underlying buffer.
    pub fn as_slice(&self) -> &[f32] {
        self.buffer.as_ref().unwrap()
    }

    /// Get a mutable reference to the underlying buffer.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.buffer.as_mut().unwrap()
    }

    /// Take ownership of the buffer (it won't be returned to the pool).
    pub fn take(mut self) -> Vec<f32> {
        self.buffer.take().unwrap()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            release_buffer(buf);
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up_pow2() {
        assert_eq!(MemoryPool::round_up_pow2(0), 1);
        assert_eq!(MemoryPool::round_up_pow2(1), 1);
        assert_eq!(MemoryPool::round_up_pow2(2), 2);
        assert_eq!(MemoryPool::round_up_pow2(3), 4);
        assert_eq!(MemoryPool::round_up_pow2(5), 8);
        assert_eq!(MemoryPool::round_up_pow2(1000), 1024);
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool = MemoryPool::new(1024 * 1024);

        // Acquire and release a buffer
        let buf = pool.acquire(100);
        assert_eq!(buf.len(), 100);
        let first_ptr = buf.as_ptr();
        pool.release(buf);

        // Acquire again - should reuse the same buffer from pool
        let buf2 = pool.acquire(100);
        assert_eq!(buf2.len(), 100);
        assert_eq!(buf2.as_ptr(), first_ptr); // Same allocation reused
    }

    #[test]
    fn test_pooled_buffer_guard() {
        clear_pool();
        assert_eq!(pool_usage(), 0);

        {
            let mut buf = PooledBuffer::new(1000);
            buf[0] = 1.0;
            buf[999] = 2.0;
            // Buffer is returned when dropped
        }

        // Buffer should be in pool now
        assert!(pool_usage() > 0);

        clear_pool();
        assert_eq!(pool_usage(), 0);
    }
}
