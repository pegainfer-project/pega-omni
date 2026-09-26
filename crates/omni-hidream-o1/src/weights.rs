//! Reading a sharded safetensors checkpoint.
//!
//! HiDream-O1 ships f32 shards; every tensor the engine keeps becomes a bf16
//! weight of the manifest, the dtype the reference runs in. Q|K|V and gate|up
//! are fused on the host at load, so the forward pass issues one GEMM for each.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use omni_kern::weights::File;
use omni_kern::weights::Host;

/// The shards a `model.safetensors.index.json` names.
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
        let index_path = dir.join("model.safetensors.index.json");
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
}
