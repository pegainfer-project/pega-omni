//! cuBLASLt algorithm selection for the DiT's GEMMs.

use std::collections::BTreeMap;
use std::os::raw::c_void;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::cublaslt::sys as lt;
use kern_manifest::types::GemmAlgo;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

use crate::config::Transformer;

pub type Shape = (usize, usize);

#[derive(Clone, Debug, Default)]
pub enum Gemms {
    #[default]
    Heuristic,
    Pinned(BTreeMap<Shape, GemmAlgo>),
    Candidates(BTreeMap<Shape, Vec<GemmAlgo>>),
}

/// The DiT's GEMM shapes, largest first.
pub fn step_shapes(cfg: &Transformer) -> Vec<Shape> {
    let h = cfg.inner_dim();
    let inter = cfg.mlp_hidden();
    let mut shapes = vec![(3 * h, h), (h, h), (2 * inter, h), (h, inter)];
    shapes.sort_by_key(|&(n, k)| std::cmp::Reverse(n * k));
    shapes.dedup();
    shapes
}

pub fn name((n, k): Shape) -> String {
    format!("gemm_{n}x{k}")
}

impl Gemms {
    pub fn launches(&self, shape: Shape) -> Vec<Value> {
        let entry = "extern:cublaslt_bf16_tn";
        match self {
            Gemms::Heuristic => vec![json!({"entry": entry})],
            Gemms::Pinned(pins) => match pins.get(&shape) {
                Some(algo) => vec![json!({"entry": entry, "algo": algo})],
                None => vec![json!({"entry": entry})],
            },
            Gemms::Candidates(all) => match all.get(&shape) {
                Some(algos) => (1..)
                    .zip(algos)
                    .map(|(i, algo)| json!({"entry": entry, "algo": algo, "when": {"var": name(shape), "min": i, "max": i}}))
                    .collect(),
                None => vec![json!({"entry": entry})],
            },
        }
    }

    pub fn vars(&self) -> BTreeMap<String, Value> {
        match self {
            Gemms::Candidates(all) => all.iter().map(|(&s, a)| (name(s), json!({"max": a.len()}))).collect(),
            _ => BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Device {
    pub gpu: String,
    pub cublaslt: usize,
}

impl Device {
    pub fn of(ordinal: usize) -> Result<Device> {
        let ctx = cudarc::driver::CudaContext::new(ordinal).context("CUDA context")?;
        Ok(Device { gpu: ctx.name()?, cublaslt: unsafe { lt::cublasLtGetVersion() } })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pin {
    pub n: usize,
    pub k: usize,
    pub algo: GemmAlgo,
    pub step_ms: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pins {
    pub device: Device,
    pub gemms: Vec<Pin>,
}

impl Pins {
    pub fn load(path: &Path, ordinal: usize) -> Result<Gemms> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let pins: Pins = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let here = Device::of(ordinal)?;
        ensure!(
            pins.device == here,
            "{} was measured on {:?}, this is {here:?}: tune again",
            path.display(),
            pins.device
        );
        Ok(Gemms::Pinned(pins.gemms.into_iter().map(|p| ((p.n, p.k), p.algo)).collect()))
    }
}

pub fn candidates(ordinal: usize, m: usize, (n, k): Shape) -> Result<Vec<GemmAlgo>> {
    use lt::cublasLtMatmulAlgoConfigAttributes_t as Cfg;
    let ctx = cudarc::driver::CudaContext::new(ordinal)?;
    ctx.bind_to_thread()?;
    let ok = |s: lt::cublasStatus_t, what: &str| {
        if s == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { bail!("cublasLt {what}: {s:?}") }
    };
    let bf = lt::cudaDataType::CUDA_R_16BF;
    let mut found = vec![unsafe { std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>() }; 32];
    let mut count = 0;
    unsafe {
        let mut handle = std::ptr::null_mut();
        ok(lt::cublasLtCreate(&mut handle), "create")?;
        let mut desc = std::ptr::null_mut();
        ok(
            lt::cublasLtMatmulDescCreate(
                &mut desc,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt::cudaDataType::CUDA_R_32F,
            ),
            "descriptor",
        )?;
        let t = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
        ok(
            lt::cublasLtMatmulDescSetAttribute(
                desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &t as *const _ as *const c_void,
                4,
            ),
            "transa",
        )?;
        let mut layouts = [std::ptr::null_mut(); 3];
        for (l, (rows, cols)) in layouts.iter_mut().zip([(k, n), (k, m), (n, m)]) {
            ok(lt::cublasLtMatrixLayoutCreate(l, bf, rows as u64, cols as u64, rows as i64), "layout")?;
        }
        let mut pref = std::ptr::null_mut();
        ok(lt::cublasLtMatmulPreferenceCreate(&mut pref), "preference")?;
        let ws: usize = 32 << 20;
        let mask = lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as u32;
        let set = |attr, v: *const c_void, size| {
            ok(lt::cublasLtMatmulPreferenceSetAttribute(pref, attr, v, size), "preference")
        };
        set(
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws as *const _ as _,
            8,
        )?;
        set(
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &mask as *const _ as _,
            4,
        )?;
        let [w, a, c] = layouts;
        let heuristic =
            lt::cublasLtMatmulAlgoGetHeuristic(handle, desc, w, a, c, c, pref, 32, found.as_mut_ptr(), &mut count);
        lt::cublasLtMatmulPreferenceDestroy(pref);
        for l in layouts {
            lt::cublasLtMatrixLayoutDestroy(l);
        }
        lt::cublasLtMatmulDescDestroy(desc);
        lt::cublasLtDestroy(handle);
        ok(heuristic, "heuristic")?;
    }
    let get = |algo: &lt::cublasLtMatmulAlgo_t, attr: Cfg, size: usize| {
        let (mut v, mut written) = (0u64, 0usize);
        unsafe {
            lt::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, &mut v as *mut _ as *mut c_void, size, &mut written)
        };
        v
    };
    let mut out: Vec<GemmAlgo> = Vec::new();
    for r in &found[..count as usize] {
        if r.state != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            continue;
        }
        let a = &r.algo;
        let algo = GemmAlgo {
            id: get(a, Cfg::CUBLASLT_ALGO_CONFIG_ID, 4) as i32,
            tile: get(a, Cfg::CUBLASLT_ALGO_CONFIG_TILE_ID, 4) as u32,
            stages: get(a, Cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID, 4) as u32,
            split_k: get(a, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 4) as i32,
            reduction: get(a, Cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 4) as u32,
            swizzle: get(a, Cfg::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 4) as u32,
            custom: get(a, Cfg::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 4) as u32,
            inner_shape: get(a, Cfg::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 2) as u16,
            cluster_shape: get(a, Cfg::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 2) as u16,
        };
        if algo.split_k <= 1 && !out.contains(&algo) {
            out.push(algo);
        }
    }
    Ok(out)
}
