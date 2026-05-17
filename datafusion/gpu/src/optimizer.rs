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

//! [`InsertGpuExec`] — physical optimizer rule that rewrites supported CPU
//! subtrees into GPU operators with [`crate::boundary::GpuUploadExec`] /
//! [`crate::boundary::GpuDownloadExec`] at the boundaries.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_optimizer::optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;

/// Inserts GPU operators into a physical plan.
///
/// In the v1 scaffold this rule is a no-op: it returns the plan unchanged.
/// The pattern-matching logic that swaps `ProjectionExec` / `FilterExec`
/// subtrees for their GPU equivalents lands once the CUDA kernels are wired
/// up. The rule is shipped now so it can be registered against a
/// `SessionStateBuilder` end-to-end and so its position in the rule order can
/// be validated against the rest of the physical optimizer pipeline.
#[derive(Debug, Default)]
pub struct InsertGpuExec;

impl InsertGpuExec {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for InsertGpuExec {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // TODO(gpu-v1): walk the plan tree; for maximal subtrees consisting
        // only of `ProjectionExec` and `FilterExec` over a non-GPU leaf,
        // produce a GPU subtree of `GpuProjectionExec`/`GpuFilterExec` rooted
        // under a `GpuDownloadExec` and bottomed by a `GpuUploadExec` wrapping
        // the original leaf.
        Ok(plan)
    }

    fn name(&self) -> &str {
        "InsertGpuExec"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
