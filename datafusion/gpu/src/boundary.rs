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

//! Boundary operators that move data across the host/device line.
//!
//! [`GpuUploadExec`] consumes a CPU [`ExecutionPlan`] child and produces a
//! stream of device-resident [`crate::GpuRecordBatch`]es.
//! [`GpuDownloadExec`] is the inverse — it consumes a
//! [`crate::GpuExecutionPlan`] child and produces a normal CPU
//! `RecordBatchStream`.
//!
//! These are the *only* places where host and device data legally meet. The
//! [`crate::optimizer::InsertGpuExec`] rule is responsible for inserting them
//! at the edges of every GPU subtree it creates.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{
    Boundedness, EmissionType, PlanProperties,
};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, SendableRecordBatchStream,
};

use crate::execution_plan::{GpuExecutionPlan, as_gpu_execution_plan};
use crate::stream::SendableGpuRecordBatchStream;

fn passthrough_plan_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(1),
        EmissionType::Incremental,
        Boundedness::Bounded,
    ))
}

/// CPU → GPU boundary. Pulls host [`RecordBatch`]es from `child`, copies their
/// buffers to device memory, and emits [`crate::GpuRecordBatch`]es.
#[derive(Debug)]
pub struct GpuUploadExec {
    child: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl GpuUploadExec {
    pub fn new(child: Arc<dyn ExecutionPlan>) -> Self {
        let schema = child.schema();
        let properties = passthrough_plan_properties(Arc::clone(&schema));
        Self {
            child,
            schema,
            properties,
        }
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.child
    }
}

impl DisplayAs for GpuUploadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuUploadExec")
    }
}

impl ExecutionPlan for GpuUploadExec {
    fn name(&self) -> &str {
        "GpuUploadExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("GpuUploadExec must have exactly one child");
        }
        let [child] = children.try_into().unwrap();
        Ok(Arc::new(GpuUploadExec::new(child)))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        internal_err!(
            "GpuUploadExec produces device-resident batches; call gpu_execute or \
             wrap in GpuDownloadExec before consuming on the host"
        )
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

impl GpuExecutionPlan for GpuUploadExec {
    fn gpu_execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableGpuRecordBatchStream> {
        // TODO(gpu-v1): pull host batches from `self.child.execute(...)`, copy
        // each column's `Buffer` into a `GpuBuffer` via `cudarc`, account the
        // allocation against the task's `GpuMemoryPool`, and emit
        // `GpuRecordBatch`es over a `GpuRecordBatchStream` adapter.
        unimplemented!("GpuUploadExec::gpu_execute: CUDA backend not yet implemented")
    }
}

/// GPU → CPU boundary. Pulls device batches from a [`GpuExecutionPlan`] child
/// and copies them back into host [`RecordBatch`]es.
///
/// Stores the child twice: once typed as `Arc<dyn GpuExecutionPlan>` (so
/// `gpu_execute` can be called) and once upcast to `Arc<dyn ExecutionPlan>`
/// (so the framework can borrow it via [`ExecutionPlan::children`]). Both
/// `Arc`s point to the same allocation.
pub struct GpuDownloadExec {
    gpu_child: Arc<dyn GpuExecutionPlan>,
    child_as_exec: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl GpuDownloadExec {
    pub fn new(gpu_child: Arc<dyn GpuExecutionPlan>) -> Self {
        let gpu_child_clone: Arc<dyn GpuExecutionPlan> = Arc::clone(&gpu_child);
        // Trait-upcast `Arc<dyn GpuExecutionPlan>` → `Arc<dyn ExecutionPlan>`
        // (stable since Rust 1.86); same allocation, different vtable view.
        let child_as_exec: Arc<dyn ExecutionPlan> = gpu_child_clone;
        let schema = child_as_exec.schema();
        let properties = passthrough_plan_properties(Arc::clone(&schema));
        Self {
            gpu_child,
            child_as_exec,
            schema,
            properties,
        }
    }

    pub fn gpu_input(&self) -> &Arc<dyn GpuExecutionPlan> {
        &self.gpu_child
    }
}

impl fmt::Debug for GpuDownloadExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuDownloadExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl DisplayAs for GpuDownloadExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuDownloadExec")
    }
}

impl ExecutionPlan for GpuDownloadExec {
    fn name(&self) -> &str {
        "GpuDownloadExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child_as_exec]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("GpuDownloadExec must have exactly one child");
        }
        let [new_child] = children.try_into().unwrap();
        if Arc::ptr_eq(&new_child, &self.child_as_exec) {
            return Ok(self);
        }
        // The framework gave us a rewritten child. We need it typed as
        // `Arc<dyn GpuExecutionPlan>` to plug into a new `GpuDownloadExec`.
        // Recover that by downcasting against the closed set of GPU operator
        // types and re-typing the new child via its constructor pattern. For
        // v1 we only support the no-op case (pointer-equal child) — any
        // other replacement would require materialising a typed Arc, which
        // in turn requires each GPU op to be `Clone`. Defer.
        let _typed: &dyn GpuExecutionPlan = as_gpu_execution_plan(&new_child)
            .ok_or_else(|| {
                datafusion_common::DataFusionError::Internal(
                    "GpuDownloadExec: new child is not a known GpuExecutionPlan".into(),
                )
            })?;
        internal_err!(
            "GpuDownloadExec::with_new_children: rewriting the GPU child is not yet \
             supported (see TODO in datafusion-gpu/src/boundary.rs)"
        )
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // TODO(gpu-v1): call `self.gpu_child.gpu_execute(...)`, copy each
        // emitted `GpuRecordBatch` from device memory back into host buffers,
        // and yield `RecordBatch`es via a `RecordBatchStreamAdapter`.
        unimplemented!("GpuDownloadExec::execute: CUDA backend not yet implemented")
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

// Force `Any` to be in scope so `as_any()` calls in this module resolve
// against the blanket impl when needed.
const _: fn() = || {
    fn assert_any<T: Any>() {}
    assert_any::<GpuUploadExec>();
    assert_any::<GpuDownloadExec>();
};
