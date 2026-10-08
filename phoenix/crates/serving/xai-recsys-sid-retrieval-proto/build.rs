// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("proto/sid_retrieval.proto")?;
    Ok(())
}
