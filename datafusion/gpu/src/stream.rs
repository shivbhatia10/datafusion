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

//! Async stream of [`GpuRecordBatch`]es. Parallel to `RecordBatchStream` in
//! `datafusion-execution`, intentionally a distinct type so device-resident
//! batches cannot accidentally flow into CPU operators.

use std::pin::Pin;

use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use futures::Stream;

use crate::batch::GpuRecordBatch;

/// Async stream of device-resident batches with a known schema.
pub trait GpuRecordBatchStream: Stream<Item = Result<GpuRecordBatch>> {
    fn schema(&self) -> SchemaRef;
}

/// A pinned, sendable [`GpuRecordBatchStream`]. The return type of
/// [`crate::GpuExecutionPlan::gpu_execute`].
pub type SendableGpuRecordBatchStream = Pin<Box<dyn GpuRecordBatchStream + Send>>;
