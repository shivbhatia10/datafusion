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

//! Device-memory accounting. Mirrors the shape of
//! `datafusion_execution::memory_pool::{MemoryPool, MemoryConsumer,
//! MemoryReservation}` but tracks GPU memory separately from host memory.

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion_common::{DataFusionError, Result};

/// Device-memory pool. Implementations decide whether and how to enforce a
/// limit.
pub trait GpuMemoryPool: Send + Sync + Debug {
    fn name(&self) -> &str;

    /// Attempt to grow `reservation` by `additional` bytes. On failure the
    /// reservation must be unchanged.
    fn try_grow(
        &self,
        reservation: &GpuMemoryReservation,
        additional: usize,
    ) -> Result<()>;

    /// Release `bytes` from `reservation`. Must be infallible.
    fn shrink(&self, reservation: &GpuMemoryReservation, bytes: usize);

    /// Total bytes currently reserved across all consumers.
    fn reserved(&self) -> usize;
}

/// Named handle used to identify an allocator for diagnostics
/// (e.g. `"GpuHashJoin:partition-3:build"`).
#[derive(Debug)]
pub struct GpuMemoryConsumer {
    name: String,
}

impl GpuMemoryConsumer {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn register(self, pool: Arc<dyn GpuMemoryPool>) -> GpuMemoryReservation {
        GpuMemoryReservation {
            consumer: self,
            pool,
            size: AtomicUsize::new(0),
        }
    }
}

/// A unit of device-memory reservation. Dropping the reservation releases its
/// remaining bytes back to the pool.
#[derive(Debug)]
pub struct GpuMemoryReservation {
    consumer: GpuMemoryConsumer,
    pool: Arc<dyn GpuMemoryPool>,
    size: AtomicUsize,
}

impl GpuMemoryReservation {
    pub fn try_grow(&self, additional: usize) -> Result<()> {
        self.pool.try_grow(self, additional)?;
        self.size.fetch_add(additional, Ordering::SeqCst);
        Ok(())
    }

    pub fn shrink(&self, bytes: usize) {
        let cur = self.size.load(Ordering::SeqCst);
        let to_release = bytes.min(cur);
        self.pool.shrink(self, to_release);
        self.size.fetch_sub(to_release, Ordering::SeqCst);
    }

    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    pub fn consumer(&self) -> &GpuMemoryConsumer {
        &self.consumer
    }
}

impl Drop for GpuMemoryReservation {
    fn drop(&mut self) {
        let remaining = self.size.load(Ordering::SeqCst);
        if remaining > 0 {
            self.pool.shrink(self, remaining);
        }
    }
}

/// Pool with no upper limit; tracks usage for diagnostics only. Useful for
/// tests and for the v1 scaffold before a CUDA-backed pool exists.
#[derive(Debug, Default)]
pub struct UnboundedGpuMemoryPool {
    reserved: AtomicUsize,
}

impl UnboundedGpuMemoryPool {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GpuMemoryPool for UnboundedGpuMemoryPool {
    fn name(&self) -> &str {
        "UnboundedGpuMemoryPool"
    }

    fn try_grow(
        &self,
        _reservation: &GpuMemoryReservation,
        additional: usize,
    ) -> Result<()> {
        self.reserved.fetch_add(additional, Ordering::SeqCst);
        Ok(())
    }

    fn shrink(&self, _reservation: &GpuMemoryReservation, bytes: usize) {
        self.reserved.fetch_sub(bytes, Ordering::SeqCst);
    }

    fn reserved(&self) -> usize {
        self.reserved.load(Ordering::SeqCst)
    }
}

/// Bounded pool with a hard byte cap; `try_grow` returns
/// [`DataFusionError::ResourcesExhausted`] when the cap would be exceeded.
#[derive(Debug)]
pub struct BoundedGpuMemoryPool {
    limit: usize,
    reserved: AtomicUsize,
}

impl BoundedGpuMemoryPool {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            reserved: AtomicUsize::new(0),
        }
    }

    pub fn limit(&self) -> usize {
        self.limit
    }
}

impl GpuMemoryPool for BoundedGpuMemoryPool {
    fn name(&self) -> &str {
        "BoundedGpuMemoryPool"
    }

    fn try_grow(
        &self,
        reservation: &GpuMemoryReservation,
        additional: usize,
    ) -> Result<()> {
        let prev = self.reserved.fetch_add(additional, Ordering::SeqCst);
        if prev + additional > self.limit {
            self.reserved.fetch_sub(additional, Ordering::SeqCst);
            return Err(DataFusionError::ResourcesExhausted(format!(
                "GPU memory limit exceeded: consumer {:?} requested {} bytes \
                 but pool {} has only {} bytes free (limit {})",
                reservation.consumer().name(),
                additional,
                self.name(),
                self.limit.saturating_sub(prev),
                self.limit,
            )));
        }
        Ok(())
    }

    fn shrink(&self, _reservation: &GpuMemoryReservation, bytes: usize) {
        self.reserved.fetch_sub(bytes, Ordering::SeqCst);
    }

    fn reserved(&self) -> usize {
        self.reserved.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_releases_on_drop() {
        let pool: Arc<dyn GpuMemoryPool> = Arc::new(UnboundedGpuMemoryPool::new());
        {
            let reservation = GpuMemoryConsumer::new("test").register(Arc::clone(&pool));
            reservation.try_grow(1024).unwrap();
            assert_eq!(pool.reserved(), 1024);
        }
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn bounded_pool_rejects_over_limit() {
        let pool: Arc<dyn GpuMemoryPool> = Arc::new(BoundedGpuMemoryPool::new(1024));
        let reservation = GpuMemoryConsumer::new("test").register(Arc::clone(&pool));
        reservation.try_grow(512).unwrap();
        let err = reservation.try_grow(1024).unwrap_err();
        assert!(matches!(err, DataFusionError::ResourcesExhausted(_)));
        assert_eq!(reservation.size(), 512);
        assert_eq!(pool.reserved(), 512);
    }
}
