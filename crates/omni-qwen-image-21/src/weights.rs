//! Reading a sharded safetensors checkpoint.
//!
//! Qwen-Image-2.1 ships bf16 shards; every tensor the engine keeps becomes a
//! bf16 weight of the manifest, the dtype the reference runs in. Q|K|V and
//! gate|up are fused on the host at load, so the forward pass issues one GEMM
//! for each.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use omni_kern::weights::File;
use omni_kern::weights::Host;
use omni_kern::weights::concat_rows;

/// The index a checkpoint names its shards in, in the order they are tried.
const INDEXES: [&str; 2] = ["model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"];

/// The shards a safetensors index names.
pub struct Checkpoint {
    shards: Vec<File>,
    owner: BTreeMap<String, usize>,
}

#[derive(serde::Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let index_path = INDEXES
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.exists())
            .with_context(|| format!("no safetensors index in {} (tried {})", dir.display(), INDEXES.join(", ")))?;
        let raw = std::fs::read_to_string(&index_path).with_context(|| format!("reading {}", index_path.display()))?;
        let index: Index = serde_json::from_str(&raw).with_context(|| format!("parsing {}", index_path.display()))?;
        let files: Vec<String> = index.weight_map.values().cloned().collect::<BTreeSet<_>>().into_iter().collect();
        let shards = files.iter().map(|f| File::open(&dir.join(f))).collect::<Result<Vec<_>>>()?;
        let owner = index
            .weight_map
            .into_iter()
            .map(|(name, file)| (name, files.iter().position(|f| *f == file).expect("file listed")))
            .collect();
        Ok(Self { shards, owner })
    }

    /// The tensor `name`, which must have exactly `shape`.
    pub fn expect(&self, name: &str, shape: &[usize]) -> Result<Host> {
        let shard = self.owner.get(name).with_context(|| format!("tensor {name} missing"))?;
        self.shards[*shard].expect(name, shape)
    }

    /// The tensors `names`, each of the matrix shape `shape`, row-concatenated
    /// into one `[names.len() * shape[0], shape[1]]` weight: Q|K|V and gate|up.
    pub fn fuse(&self, names: &[&str], shape: &[usize]) -> Result<Host> {
        ensure!(!names.is_empty(), "fusing no tensors");
        ensure!(shape.len() == 2, "fusing needs a matrix shape, got {shape:?}");
        let parts = names.iter().map(|name| self.expect(name, shape)).collect::<Result<Vec<_>>>()?;
        Ok(Host { shape: vec![shape[0] * parts.len(), shape[1]], data: concat_rows(&parts) })
    }
}
