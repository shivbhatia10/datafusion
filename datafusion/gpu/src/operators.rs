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

//! v1 GPU operator stubs: [`GpuProjectionExec`] and [`GpuFilterExec`].
//!
//! Both store a `GpuExecutionPlan` child (typed) and an upcast
//! `Arc<dyn ExecutionPlan>` view of the same allocation so the framework can
//! borrow children via `ExecutionPlan::children`. Bodies are stubbed pending
//! the CUDA kernel work.

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

use crate::execution_plan::GpuExecutionPlan;
use crate::stream::SendableGpuRecordBatchStream;

fn projection_plan_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(1),
        EmissionType::Incremental,
        Boundedness::Bounded,
    ))
}

/// GPU equivalent of `ProjectionExec`. v1 plan: column passthrough where the
/// expression is `Column`; arithmetic via a hand-rolled CUDA kernel.
pub struct GpuProjectionExec {
    expressions: Vec<(Arc<dyn PhysicalExpr>, String)>,
    gpu_input: Arc<dyn GpuExecutionPlan>,
    input_as_exec: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl GpuProjectionExec {
    pub fn try_new(
        expressions: Vec<(Arc<dyn PhysicalExpr>, String)>,
        gpu_input: Arc<dyn GpuExecutionPlan>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let gpu_input_clone: Arc<dyn GpuExecutionPlan> = Arc::clone(&gpu_input);
        let input_as_exec: Arc<dyn ExecutionPlan> = gpu_input_clone;
        let properties = projection_plan_properties(Arc::clone(&schema));
        Ok(Self {
            expressions,
            gpu_input,
            input_as_exec,
            schema,
            properties,
        })
    }

    pub fn expressions(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.expressions
    }

    pub fn gpu_input(&self) -> &Arc<dyn GpuExecutionPlan> {
        &self.gpu_input
    }
}

impl fmt::Debug for GpuProjectionExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuProjectionExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl DisplayAs for GpuProjectionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuProjectionExec")
    }
}

impl ExecutionPlan for GpuProjectionExec {
    fn name(&self) -> &str {
        "GpuProjectionExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input_as_exec]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("GpuProjectionExec must have exactly one child");
        }
        let [new_child] = children.try_into().unwrap();
        if Arc::ptr_eq(&new_child, &self.input_as_exec) {
            return Ok(self);
        }
        internal_err!(
            "GpuProjectionExec::with_new_children: replacing the GPU input is not \
             yet supported in v1 scaffold"
        )
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        internal_err!(
            "GpuProjectionExec produces device-resident batches; wrap in \
             GpuDownloadExec before consuming on the host"
        )
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let mut tnr = TreeNodeRecursion::Continue;
        for (expr, _alias) in &self.expressions {
            tnr = tnr.visit_sibling(|| f(expr.as_ref()))?;
        }
        Ok(tnr)
    }
}

impl GpuExecutionPlan for GpuProjectionExec {
    fn gpu_execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableGpuRecordBatchStream> {
        // TODO(gpu-v1): walk `self.expressions`; for each `Column`, return the
        // input column unchanged; for arithmetic, dispatch to the appropriate
        // CUDA kernel; assemble a new `GpuRecordBatch` against the projected
        // schema.
        unimplemented!("GpuProjectionExec::gpu_execute: CUDA kernels not yet implemented")
    }
}

/// GPU equivalent of `FilterExec`. v1 plan: predicate is evaluated as a
/// boolean mask kernel, then a gather kernel materialises the surviving rows.
pub struct GpuFilterExec {
    predicate: Arc<dyn PhysicalExpr>,
    gpu_input: Arc<dyn GpuExecutionPlan>,
    input_as_exec: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl GpuFilterExec {
    pub fn try_new(
        predicate: Arc<dyn PhysicalExpr>,
        gpu_input: Arc<dyn GpuExecutionPlan>,
    ) -> Result<Self> {
        let gpu_input_clone: Arc<dyn GpuExecutionPlan> = Arc::clone(&gpu_input);
        let input_as_exec: Arc<dyn ExecutionPlan> = gpu_input_clone;
        let schema = input_as_exec.schema();
        let properties = projection_plan_properties(Arc::clone(&schema));
        Ok(Self {
            predicate,
            gpu_input,
            input_as_exec,
            schema,
            properties,
        })
    }

    pub fn predicate(&self) -> &Arc<dyn PhysicalExpr> {
        &self.predicate
    }

    pub fn gpu_input(&self) -> &Arc<dyn GpuExecutionPlan> {
        &self.gpu_input
    }
}

impl fmt::Debug for GpuFilterExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuFilterExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl DisplayAs for GpuFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GpuFilterExec")
    }
}

impl ExecutionPlan for GpuFilterExec {
    fn name(&self) -> &str {
        "GpuFilterExec"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input_as_exec]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("GpuFilterExec must have exactly one child");
        }
        let [new_child] = children.try_into().unwrap();
        if Arc::ptr_eq(&new_child, &self.input_as_exec) {
            return Ok(self);
        }
        internal_err!(
            "GpuFilterExec::with_new_children: replacing the GPU input is not yet \
             supported in v1 scaffold"
        )
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        internal_err!(
            "GpuFilterExec produces device-resident batches; wrap in \
             GpuDownloadExec before consuming on the host"
        )
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        f(self.predicate.as_ref())?;
        Ok(TreeNodeRecursion::Continue)
    }
}

impl GpuExecutionPlan for GpuFilterExec {
    fn gpu_execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableGpuRecordBatchStream> {
        // TODO(gpu-v1): evaluate `self.predicate` against the input batch on
        // device, producing a boolean selection mask; gather surviving rows
        // into a new `GpuRecordBatch`.
        unimplemented!("GpuFilterExec::gpu_execute: CUDA kernels not yet implemented")
    }
}
