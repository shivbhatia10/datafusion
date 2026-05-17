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

//! Device-resident column buffers and record batches.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};

use crate::memory_pool::GpuMemoryReservation;

/// A column buffer resident in GPU device memory.
///
/// In this scaffold the underlying storage is a host-side placeholder; the
/// real implementation will replace [`Backing::HostPlaceholder`] with a CUDA
/// device allocation (e.g. `cudarc::driver::CudaSlice`). Operators must
/// treat the bytes as opaque — they are *not* a valid host pointer for kernel
/// inputs.
#[derive(Debug)]
pub struct GpuBuffer {
    byte_len: usize,
    #[expect(dead_code, reason = "read once the CUDA backend lands")]
    backing: Backing,
}

#[derive(Debug)]
enum Backing {
    /// Host-resident placeholder. Held only so the scaffold compiles and
    /// round-trips data through `GpuUploadExec`/`GpuDownloadExec` once those
    /// land; never to be dereferenced by GPU code.
    HostPlaceholder(
        #[expect(dead_code, reason = "read once the CUDA backend lands")] Vec<u8>,
    ),
}

impl GpuBuffer {
    pub fn from_host_placeholder(bytes: Vec<u8>) -> Self {
        Self {
            byte_len: bytes.len(),
            backing: Backing::HostPlaceholder(bytes),
        }
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// Arrow-shaped batch whose column buffers live in device memory.
///
/// This is the unit of data exchanged between operators implementing
/// [`crate::GpuExecutionPlan`]. It deliberately does *not* implement any
/// conversion to `arrow::record_batch::RecordBatch`; crossing the host/device
/// boundary must go through [`crate::boundary::GpuUploadExec`] /
/// [`crate::boundary::GpuDownloadExec`].
#[derive(Debug)]
pub struct GpuRecordBatch {
    schema: SchemaRef,
    columns: Vec<Arc<GpuBuffer>>,
    num_rows: usize,
    // Dropped with the batch, releasing the device-memory reservation.
    _reservation: GpuMemoryReservation,
}

impl GpuRecordBatch {
    pub fn try_new(
        schema: SchemaRef,
        columns: Vec<Arc<GpuBuffer>>,
        num_rows: usize,
        reservation: GpuMemoryReservation,
    ) -> Result<Self> {
        if schema.fields().len() != columns.len() {
            return internal_err!(
                "GpuRecordBatch: schema has {} fields but {} columns supplied",
                schema.fields().len(),
                columns.len()
            );
        }
        Ok(Self {
            schema,
            columns,
            num_rows,
            _reservation: reservation,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, idx: usize) -> &Arc<GpuBuffer> {
        &self.columns[idx]
    }

    pub fn columns(&self) -> &[Arc<GpuBuffer>] {
        &self.columns
    }
}
