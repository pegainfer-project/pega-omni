//! Reading a sharded safetensors checkpoint.
//!
//! HiDream-O1 ships f32 shards; every tensor the engine keeps becomes a bf16
//! weight of the manifest, the dtype the reference runs in. Q|K|V and gate|up
//! are fused on the host at load, so the forward pass issues one GEMM for each.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::Metadata;

/// A host tensor: shape and values.
pub struct Host {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

struct Shard {
    map: Mmap,
    start: usize,
    meta: Metadata,
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let (header, meta) = SafeTensors::read_metadata(&map)?;
        Ok(Self { map, start: 8 + header, meta })
    }
}

/// One safetensors file, or the shards a `model.safetensors.index.json` names.
pub struct Checkpoint {
    shards: Vec<Shard>,
    owner: BTreeMap<String, usize>,
}

#[derive(serde::Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_file() {
            let shard = Shard::open(path)?;
            let owner = shard.meta.tensors().into_keys().map(|name| (name, 0)).collect();
            return Ok(Self { shards: vec![shard], owner });
        }
        let index_path = path.join("model.safetensors.index.json");
        let raw = std::fs::read_to_string(&index_path).with_context(|| format!("reading {}", index_path.display()))?;
        let index: Index = serde_json::from_str(&raw).with_context(|| format!("parsing {}", index_path.display()))?;
        let files: Vec<String> =
            index.weight_map.values().cloned().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let shards = files.iter().map(|f| Shard::open(&path.join(f))).collect::<Result<Vec<_>>>()?;
        let owner = index
            .weight_map
            .into_iter()
            .map(|(name, file)| (name, files.iter().position(|f| *f == file).expect("file listed")))
            .collect();
        Ok(Self { shards, owner })
    }

    fn raw(&self, name: &str) -> Result<(Dtype, Vec<usize>, &[u8])> {
        let shard = &self.shards[*self.owner.get(name).with_context(|| format!("tensor {name} missing"))?];
        let info = shard.meta.info(name).with_context(|| format!("tensor {name} missing from its shard"))?;
        let (a, b) = info.data_offsets;
        Ok((info.dtype, info.shape.clone(), &shard.map[shard.start + a..shard.start + b]))
    }

    pub fn host(&self, name: &str) -> Result<Host> {
        let (dtype, shape, bytes) = self.raw(name)?;
        let data = match dtype {
            Dtype::F32 => bytes.as_chunks().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
            Dtype::BF16 => bytes.as_chunks().0.iter().map(|&b| bf16::from_le_bytes(b).to_f32()).collect(),
            other => bail!("tensor {name}: unsupported dtype {other:?}"),
        };
        Ok(Host { shape, data })
    }

    pub fn ints(&self, name: &str) -> Result<Vec<i32>> {
        let (dtype, _, bytes) = self.raw(name)?;
        match dtype {
            Dtype::I64 => Ok(bytes.as_chunks().0.iter().map(|&b| i64::from_le_bytes(b) as i32).collect()),
            Dtype::I32 => Ok(bytes.as_chunks().0.iter().map(|&b| i32::from_le_bytes(b)).collect()),
            other => bail!("tensor {name}: expected integers, got {other:?}"),
        }
    }

    /// The tensor `name`, which must have exactly `shape`.
    pub fn expect(&self, name: &str, shape: &[usize]) -> Result<Host> {
        let t = self.host(name)?;
        ensure!(t.shape == shape, "tensor {name}: shape {:?}, expected {shape:?}", t.shape);
        Ok(t)
    }
}

/// Row-major tensors stacked along their first axis.
pub fn concat_rows(parts: &[Host]) -> Vec<f32> {
    parts.iter().flat_map(|p| p.data.iter().copied()).collect()
}
