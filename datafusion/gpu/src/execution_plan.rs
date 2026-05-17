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

//! [`GpuExecutionPlan`] — subtrait of `ExecutionPlan` that produces device
//! batches from `gpu_execute`.

use std::any::Any;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::ExecutionPlan;

use crate::boundary::GpuUploadExec;
use crate::operators::{GpuFilterExec, GpuProjectionExec};
use crate::stream::SendableGpuRecordBatchStream;

/// An [`ExecutionPlan`] whose output is a stream of device-resident batches.
///
/// Every `GpuExecutionPlan` is also an `ExecutionPlan` (subtrait), which gives
/// callers free trait-upcasting and lets GPU operators sit in the regular
/// physical-plan tree. The contract is that `ExecutionPlan::execute` on a
/// `GpuExecutionPlan` is unreachable in well-formed plans: the
/// [`crate::optimizer::InsertGpuExec`] rule guarantees every `GpuExecutionPlan`
/// subtree is terminated by a [`crate::boundary::GpuDownloadExec`] that calls
/// `gpu_execute` directly.
pub trait GpuExecutionPlan: ExecutionPlan {
    fn gpu_execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableGpuRecordBatchStream>;
}

/// Borrow an `Arc<dyn ExecutionPlan>` as `&dyn GpuExecutionPlan` if its
/// concrete type is one of the GPU operators defined in this crate.
///
/// Rust trait objects don't carry subtrait vtables, so there is no built-in
/// way to recover `dyn GpuExecutionPlan`-ness from an `Arc<dyn ExecutionPlan>`
/// even though every `GpuExecutionPlan` *is* an `ExecutionPlan`. Because we
/// own every concrete `GpuExecutionPlan` implementation in this crate, an
/// exhaustive `Any`-based downcast is sound.
///
/// Callers needing an owned typed handle should hold onto the original
/// `Arc<dyn GpuExecutionPlan>` from construction; this helper only yields a
/// borrowed view.
pub fn as_gpu_execution_plan(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<&dyn GpuExecutionPlan> {
    // `ExecutionPlan: Any`, so we can upcast `&dyn ExecutionPlan` to `&dyn Any`
    // via stable trait-upcasting coercion (Rust 1.86+).
    let any: &dyn Any = plan.as_ref();
    if let Some(p) = any.downcast_ref::<GpuUploadExec>() {
        return Some(p);
    }
    if let Some(p) = any.downcast_ref::<GpuProjectionExec>() {
        return Some(p);
    }
    if let Some(p) = any.downcast_ref::<GpuFilterExec>() {
        return Some(p);
    }
    None
}
