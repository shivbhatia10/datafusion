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

//! Experimental GPU physical operators for DataFusion.
//!
//! This crate scaffolds a path for DataFusion logical plans to compile down
//! to physical operators that execute on Nvidia GPUs. Data flowing between
//! GPU operators lives in device memory as [`GpuRecordBatch`]es; the only
//! places where host and device data meet are the [`GpuUploadExec`] and
//! [`GpuDownloadExec`] boundary operators.
//!
//! The crate is structured around three contracts:
//!
//! * [`GpuRecordBatch`] / [`GpuRecordBatchStream`] — the device-resident
//!   equivalents of `RecordBatch` and `RecordBatchStream`.
//! * [`GpuExecutionPlan`] — subtrait of `ExecutionPlan` whose `gpu_execute`
//!   produces a stream of `GpuRecordBatch`es.
//! * [`GpuMemoryPool`] — device-memory accounting analogous to the host
//!   `MemoryPool` in `datafusion-execution`.
//!
//! The [`InsertGpuExec`] physical optimizer rule rewrites CPU subtrees into
//! GPU ones and inserts the boundary operators automatically.
//!
//! [`GpuUploadExec`]: boundary::GpuUploadExec
//! [`GpuDownloadExec`]: boundary::GpuDownloadExec
//! [`InsertGpuExec`]: optimizer::InsertGpuExec

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(clippy::clone_on_ref_ptr)]
#![cfg_attr(test, allow(clippy::needless_pass_by_value))]

pub mod batch;
pub mod boundary;
pub mod execution_plan;
pub mod memory_pool;
pub mod operators;
pub mod optimizer;
pub mod stream;

pub use batch::{GpuBuffer, GpuRecordBatch};
pub use execution_plan::{GpuExecutionPlan, as_gpu_execution_plan};
pub use memory_pool::{
    GpuMemoryConsumer, GpuMemoryPool, GpuMemoryReservation, UnboundedGpuMemoryPool,
};
pub use optimizer::InsertGpuExec;
pub use stream::{GpuRecordBatchStream, SendableGpuRecordBatchStream};
