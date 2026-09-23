//! The codec decoder: 16-codebook frames to 24 kHz samples, 1920 per frame.
//!
//! RVQ lookup → causal conv → sliding-window transformer → two ×2 upsamplers
//! (transposed conv + ConvNeXt) → four SnakeBeta/transposed-conv blocks
//! (×8, ×5, ×4, ×3) with dilated residual units → SnakeBeta, conv, clamp.
//! Every layer is causal, so a chunk decodes from its frames plus some left
//! context and keeps only the samples past the context, which is how the
//! official implementation streams too.
//!
//! Activations are `[T, C]` rows; every dense conv is `im2col` + GEMM, every
//! transposed conv GEMM + `col2im`. Scales and projections are folded at load:
//! LayerScale into the projections it scales, ConvNeXt's gamma into its last
//! linear, the two RVQ output projections into one GEMM.

use anyhow::Result;
use anyhow::ensure;
use omni_cuda::Act;
use omni_cuda::Buf;
use omni_cuda::Gpu;
use omni_cuda::Ptr;
use omni_cuda::bf16;

use crate::config;
use crate::weights::File;
use crate::weights::Host;
use crate::weights::concat_rows;
use crate::weights::conv_taps;
use crate::weights::scale_rows;
use crate::weights::transposed_taps;
use crate::weights::upload;

const GROUPS: usize = 16;
const KERNEL: usize = 7;

/// SnakeBeta parameters, folded to `a = exp(alpha)`, `inv_b = 1 / (exp(beta) + 1e-9)`.
struct Snake {
    a: Buf<f32>,
    inv_b: Buf<f32>,
}

impl Snake {
    fn load(gpu: &Gpu, file: &File, prefix: &str, c: usize) -> Result<Self> {
        let alpha = file.expect(&format!("{prefix}.alpha"), &[c])?;
        let beta = file.expect(&format!("{prefix}.beta"), &[c])?;
        Ok(Self {
            a: gpu.upload(&alpha.data.iter().map(|x| x.exp()).collect::<Vec<_>>())?,
            inv_b: gpu.upload(&beta.data.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect::<Vec<_>>())?,
        })
    }

    fn ptrs(&self) -> (Ptr, Ptr) {
        (self.a.ptr(), self.inv_b.ptr())
    }
}

struct Linear {
    w: Buf<bf16>,
    b: Buf<bf16>,
}

fn load_conv(gpu: &Gpu, file: &File, prefix: &str, shape: [usize; 3]) -> Result<Linear> {
    let w = file.expect(&format!("{prefix}.weight"), &shape)?;
    Ok(Linear {
        w: upload(gpu, &conv_taps(&w))?,
        b: upload(gpu, &file.expect(&format!("{prefix}.bias"), &shape[..1])?.data)?,
    })
}

fn load_transposed(gpu: &Gpu, file: &File, prefix: &str, shape: [usize; 3]) -> Result<Linear> {
    let w = file.expect(&format!("{prefix}.weight"), &shape)?;
    Ok(Linear {
        w: upload(gpu, &transposed_taps(&w))?,
        b: upload(gpu, &file.expect(&format!("{prefix}.bias"), &shape[1..2])?.data)?,
    })
}

struct Layer {
    ln1: Buf<bf16>,
    qkv: Buf<bf16>,
    o: Buf<bf16>,
    ln2: Buf<bf16>,
    gate_up: Buf<bf16>,
    down: Buf<bf16>,
}

struct ConvNeXt {
    up: Linear,
    dw: Linear,
    norm: (Buf<bf16>, Buf<bf16>),
    pw1: Linear,
    pw2: Linear,
}

struct Unit {
    snake1: Snake,
    conv1: Linear,
    dilation: usize,
    snake2: Snake,
    conv2: Linear,
}

struct Block {
    snake: Snake,
    up: Linear,
    rate: usize,
    units: Vec<Unit>,
}

/// Work buffers, sized for `frames` frames per call.
struct Scratch {
    frames: usize,
    ids: Buf<i32>,
    a: Buf<bf16>,
    b: Buf<bf16>,
    c: Buf<bf16>,
    residual: Buf<bf16>,
    col: Buf<bf16>,
    wav: Buf<f32>,
}

pub struct Codec {
    cfg: config::Codec,
    samples_per_frame: usize,
    /// What the codebook pointer table indexes into.
    _codebooks: Buf<bf16>,
    t_first: Buf<u64>,
    t_rest: Buf<u64>,
    rvq_out: Buf<bf16>,
    pre_conv: Linear,
    input_proj: Linear,
    layers: Vec<Layer>,
    norm: Buf<bf16>,
    output_proj: Linear,
    upsample: Vec<ConvNeXt>,
    conv_in: Linear,
    blocks: Vec<Block>,
    snake_out: Snake,
    conv_out: Linear,
    scratch: Scratch,
}

impl Codec {
    /// Loads `speech_tokenizer/model.safetensors`; each [`Codec::decode`] call takes up to `frames` frames.
    pub fn load(gpu: &Gpu, file: &File, cfg: &config::Codec, samples_per_frame: usize, frames: usize) -> Result<Self> {
        let (dim, cb, latent, hidden) = (cfg.codebook_dim, cfg.codebook_size, cfg.latent_dim, cfg.hidden_size);
        let half = dim / 2;
        ensure!(cfg.num_quantizers == GROUPS, "{} quantizers, the engine is built for {GROUPS}", cfg.num_quantizers);
        ensure!(cfg.head_dim == 64, "codec head_dim {} unsupported (attention is built for 64)", cfg.head_dim);

        let codebook = |p: &str| -> Result<Vec<f32>> {
            let sum = file.expect(&format!("{p}._codebook.embedding_sum"), &[cb, half])?;
            let usage = file.expect(&format!("{p}._codebook.cluster_usage"), &[cb])?;
            Ok(sum
                .data
                .chunks(half)
                .zip(&usage.data)
                .flat_map(|(row, &u)| row.iter().map(move |x| x / u.max(1e-5)))
                .collect())
        };
        let books: Vec<f32> = std::iter::once(codebook("decoder.quantizer.rvq_first.vq.layers.0"))
            .chain((0..GROUPS - 1).map(|g| codebook(&format!("decoder.quantizer.rvq_rest.vq.layers.{g}"))))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let codebooks = upload(gpu, &books)?;
        let book = |g: usize| codebooks.at(g * cb * half);
        let proj = |p: &str| file.expect(&format!("decoder.quantizer.{p}.output_proj.weight"), &[dim, half, 1]);
        let (first, rest) = (proj("rvq_first")?, proj("rvq_rest")?);
        let rvq_out: Vec<f32> = (0..dim)
            .flat_map(|r| {
                first.data[r * half..(r + 1) * half]
                    .iter()
                    .chain(&rest.data[r * half..(r + 1) * half])
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();

        let pt = "decoder.pre_transformer";
        let (heads, inter) = (cfg.num_attention_heads, cfg.intermediate_size);
        let qd = heads * cfg.head_dim;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let w = |n: &str, shape: &[usize]| file.expect(&format!("{pt}.layers.{i}.{n}"), shape);
                let attn_scale = w("self_attn_layer_scale.scale", &[hidden])?;
                let mlp_scale = w("mlp_layer_scale.scale", &[hidden])?;
                Ok(Layer {
                    ln1: upload(gpu, &w("input_layernorm.weight", &[hidden])?.data)?,
                    qkv: upload(
                        gpu,
                        &concat_rows(&[
                            w("self_attn.q_proj.weight", &[qd, hidden])?,
                            w("self_attn.k_proj.weight", &[qd, hidden])?,
                            w("self_attn.v_proj.weight", &[qd, hidden])?,
                        ]),
                    )?,
                    o: upload(gpu, &scale_rows(&w("self_attn.o_proj.weight", &[hidden, qd])?.data, &attn_scale.data))?,
                    ln2: upload(gpu, &w("post_attention_layernorm.weight", &[hidden])?.data)?,
                    gate_up: upload(
                        gpu,
                        &concat_rows(&[
                            w("mlp.gate_proj.weight", &[inter, hidden])?,
                            w("mlp.up_proj.weight", &[inter, hidden])?,
                        ]),
                    )?,
                    down: upload(
                        gpu,
                        &scale_rows(&w("mlp.down_proj.weight", &[hidden, inter])?.data, &mlp_scale.data),
                    )?,
                })
            })
            .collect::<Result<_>>()?;
        let linear = |p: &str, shape: [usize; 2]| -> Result<Linear> {
            Ok(Linear {
                w: upload(gpu, &file.expect(&format!("{p}.weight"), &shape)?.data)?,
                b: upload(gpu, &file.expect(&format!("{p}.bias"), &shape[..1])?.data)?,
            })
        };

        let upsample = cfg
            .upsampling_ratios
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                let p = format!("decoder.upsample.{i}");
                let gamma = file.expect(&format!("{p}.1.gamma"), &[latent])?;
                let pw2: Host = file.expect(&format!("{p}.1.pwconv2.weight"), &[latent, 4 * latent])?;
                let pw2_b = file.expect(&format!("{p}.1.pwconv2.bias"), &[latent])?;
                Ok(ConvNeXt {
                    up: load_transposed(gpu, file, &format!("{p}.0.conv"), [latent, latent, r])?,
                    dw: Linear {
                        w: upload(gpu, &file.expect(&format!("{p}.1.dwconv.conv.weight"), &[latent, 1, KERNEL])?.data)?,
                        b: upload(gpu, &file.expect(&format!("{p}.1.dwconv.conv.bias"), &[latent])?.data)?,
                    },
                    norm: (
                        upload(gpu, &file.expect(&format!("{p}.1.norm.weight"), &[latent])?.data)?,
                        upload(gpu, &file.expect(&format!("{p}.1.norm.bias"), &[latent])?.data)?,
                    ),
                    pw1: linear(&format!("{p}.1.pwconv1"), [4 * latent, latent])?,
                    pw2: Linear {
                        w: upload(gpu, &scale_rows(&pw2.data, &gamma.data))?,
                        b: upload(gpu, &scale_rows(&pw2_b.data, &gamma.data))?,
                    },
                })
            })
            .collect::<Result<_>>()?;

        let blocks = cfg
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &rate)| {
                let p = format!("decoder.decoder.{}.block", i + 1);
                let (cin, cout) = (cfg.decoder_dim >> i, cfg.decoder_dim >> (i + 1));
                let units = [1, 3, 9]
                    .iter()
                    .enumerate()
                    .map(|(u, &dilation)| {
                        let q = format!("{p}.{}", u + 2);
                        Ok(Unit {
                            snake1: Snake::load(gpu, file, &format!("{q}.act1"), cout)?,
                            conv1: load_conv(gpu, file, &format!("{q}.conv1.conv"), [cout, cout, KERNEL])?,
                            dilation,
                            snake2: Snake::load(gpu, file, &format!("{q}.act2"), cout)?,
                            conv2: load_conv(gpu, file, &format!("{q}.conv2.conv"), [cout, cout, 1])?,
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(Block {
                    snake: Snake::load(gpu, file, &format!("{p}.0"), cin)?,
                    up: load_transposed(gpu, file, &format!("{p}.1.conv"), [cin, cout, 2 * rate])?,
                    rate,
                    units,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let out_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
        let n = cfg.upsample_rates.len() + 1;

        let codec = Self {
            t_first: gpu.upload(&[book(0)])?,
            t_rest: gpu.upload(&(1..GROUPS).map(book).collect::<Vec<_>>())?,
            _codebooks: codebooks,
            rvq_out: upload(gpu, &rvq_out)?,
            pre_conv: load_conv(gpu, file, "decoder.pre_conv.conv", [latent, dim, 3])?,
            input_proj: linear(&format!("{pt}.input_proj"), [hidden, latent])?,
            layers,
            norm: upload(gpu, &file.expect(&format!("{pt}.norm.weight"), &[hidden])?.data)?,
            output_proj: linear(&format!("{pt}.output_proj"), [latent, hidden])?,
            upsample,
            conv_in: load_conv(gpu, file, "decoder.decoder.0.conv", [cfg.decoder_dim, latent, KERNEL])?,
            blocks,
            snake_out: Snake::load(gpu, file, &format!("decoder.decoder.{n}"), out_dim)?,
            conv_out: load_conv(gpu, file, &format!("decoder.decoder.{}.conv", n + 1), [1, out_dim, KERNEL])?,
            scratch: Scratch::new(gpu, cfg, samples_per_frame, frames)?,
            cfg: cfg.clone(),
            samples_per_frame,
        };
        Ok(codec)
    }

    /// Decodes `codes` (frames × 16, frame-major) to `frames × samples_per_frame` samples in [-1, 1].
    pub fn decode(&mut self, gpu: &Gpu, codes: &[i32]) -> Result<Vec<f32>> {
        let f = codes.len() / GROUPS;
        ensure!(f <= self.scratch.frames, "{f} frames exceed the decoder's {} per call", self.scratch.frames);
        if f == 0 {
            return Ok(vec![]);
        }
        let c = &self.cfg;
        let s = &mut self.scratch;
        let (dim, half, latent, hidden) = (c.codebook_dim, c.codebook_dim / 2, c.latent_dim, c.hidden_size);
        gpu.write(&mut s.ids, 0, codes)?;
        let (a, b, x, res, col) = (s.a.ptr(), s.b.ptr(), s.c.ptr(), s.residual.ptr(), s.col.ptr());

        // RVQ: [first | Σ rest] → one projection.
        gpu.gather_sum((col, dim), false, 0, self.t_first.ptr(), (s.ids.ptr(), GROUPS, 1), f, half)?;
        gpu.gather_sum((s.col.at(half), dim), false, 0, self.t_rest.ptr(), (s.ids.at(1), GROUPS, GROUPS - 1), f, half)?;
        gpu.linear(a, col, self.rvq_out.ptr(), f, dim, dim)?;
        conv(gpu, a, &self.pre_conv, b, col, (f, dim, latent), 3)?;

        // Transformer over the frames, residual stream in `res`.
        gpu.linear(x, b, self.input_proj.w.ptr(), f, hidden, latent)?;
        gpu.bias_act(x, self.input_proj.b.ptr(), 0, x, Act::None, f, hidden)?;
        let (heads, hd, inter) = (c.num_attention_heads, c.head_dim, c.intermediate_size);
        let qkv_w = 3 * heads * hd;
        gpu.copy(res, x, f * hidden * 2)?;
        gpu.rms_norm(x, self.layers[0].ln1.ptr(), x, f, hidden, c.rms_norm_eps)?;
        let positions = s.ids.at(GROUPS * s.frames);
        for (i, l) in self.layers.iter().enumerate() {
            gpu.linear(col, x, l.qkv.ptr(), f, qkv_w, hidden)?;
            gpu.qk_norm_rope(col, qkv_w, f, (heads, heads, hd), (0, 0, 0.0), positions, (0, 0, 0), c.rope_theta)?;
            let (q, k, v) = (col, s.col.at(heads * hd), s.col.at(2 * heads * hd));
            gpu.window_prefill((q, k, v), qkv_w, a, f, heads, c.sliding_window)?;
            gpu.linear(x, a, l.o.ptr(), f, hidden, heads * hd)?;
            gpu.add_rms_norm(x, res, l.ln2.ptr(), f, hidden, c.rms_norm_eps)?;
            gpu.linear(col, x, l.gate_up.ptr(), f, 2 * inter, hidden)?;
            gpu.silu_mul(col, a, f, inter)?;
            gpu.linear(x, a, l.down.ptr(), f, hidden, inter)?;
            let next = self.layers.get(i + 1).map_or(&self.norm, |n| &n.ln1);
            gpu.add_rms_norm(x, res, next.ptr(), f, hidden, c.rms_norm_eps)?;
        }
        gpu.linear(a, x, self.output_proj.w.ptr(), f, latent, hidden)?;
        gpu.bias_act(a, self.output_proj.b.ptr(), 0, a, Act::None, f, latent)?;

        // Upsamplers: transposed conv, then ConvNeXt with its residual.
        let mut t = f;
        for (u, &r) in self.upsample.iter().zip(&c.upsampling_ratios) {
            gpu.linear(col, a, u.up.w.ptr(), t, r * latent, latent)?;
            gpu.col2im(col, u.up.b.ptr(), b, (t, latent), r, r)?;
            t *= r;
            gpu.dwconv_layernorm(
                b,
                (u.dw.w.ptr(), u.dw.b.ptr()),
                (u.norm.0.ptr(), u.norm.1.ptr()),
                x,
                (t, latent),
                KERNEL,
                1e-6,
            )?;
            gpu.linear(col, x, u.pw1.w.ptr(), t, 4 * latent, latent)?;
            gpu.bias_act(col, u.pw1.b.ptr(), 0, col, Act::Gelu, t, 4 * latent)?;
            gpu.linear(x, col, u.pw2.w.ptr(), t, latent, 4 * latent)?;
            gpu.bias_act(x, u.pw2.b.ptr(), b, a, Act::None, t, latent)?;
        }

        conv(gpu, a, &self.conv_in, b, col, (t, latent, c.decoder_dim), KERNEL)?;
        // `b` holds the block input; each block leaves its output in `b`.
        for (i, blk) in self.blocks.iter().enumerate() {
            let (cin, cout) = (c.decoder_dim >> i, c.decoder_dim >> (i + 1));
            gpu.im2col(b, 0, blk.snake.ptrs(), a, (t, cin), 1, 1)?;
            gpu.linear(col, a, blk.up.w.ptr(), t, 2 * blk.rate * cout, cin)?;
            gpu.col2im(col, blk.up.b.ptr(), b, (t, cout), 2 * blk.rate, blk.rate)?;
            t *= blk.rate;
            for u in &blk.units {
                gpu.im2col(b, 0, u.snake1.ptrs(), col, (t, cout), KERNEL, u.dilation)?;
                gpu.linear(a, col, u.conv1.w.ptr(), t, cout, KERNEL * cout)?;
                gpu.im2col(a, u.conv1.b.ptr(), u.snake2.ptrs(), x, (t, cout), 1, 1)?;
                gpu.linear(a, x, u.conv2.w.ptr(), t, cout, cout)?;
                gpu.bias_act(a, u.conv2.b.ptr(), b, b, Act::None, t, cout)?;
            }
        }
        let out_dim = c.decoder_dim >> self.blocks.len();
        gpu.conv_out(
            b,
            self.snake_out.ptrs(),
            (self.conv_out.w.ptr(), self.conv_out.b.ptr()),
            s.wav.ptr(),
            (t, out_dim),
            KERNEL,
        )?;
        debug_assert_eq!(t, f * self.samples_per_frame);
        gpu.read(s.wav.ptr(), t)
    }
}

/// `out = conv(x) + bias` for a causal conv of `k` taps: `im2col` into `col`, then one GEMM.
fn conv(
    gpu: &Gpu,
    x: Ptr,
    w: &Linear,
    out: Ptr,
    col: Ptr,
    (t, cin, cout): (usize, usize, usize),
    k: usize,
) -> Result<()> {
    gpu.im2col(x, 0, (0, 0), col, (t, cin), k, 1)?;
    gpu.linear(out, col, w.w.ptr(), t, cout, k * cin)?;
    gpu.bias_act(out, w.b.ptr(), 0, out, Act::None, t, cout)
}

impl Scratch {
    fn new(gpu: &Gpu, cfg: &config::Codec, samples_per_frame: usize, frames: usize) -> Result<Self> {
        // Per frame: the widest [T, C] activation and the widest im2col / GEMM
        // output any stage writes.
        let latent = cfg.latent_dim;
        let mut t = cfg.upsampling_ratios.iter().product::<usize>();
        let mut c = cfg.decoder_dim;
        let mut act = t * c.max(latent);
        let mut col = [
            t * KERNEL * latent,
            t * 4 * latent,
            3 * cfg.num_attention_heads * cfg.head_dim,
            2 * cfg.intermediate_size,
            3 * cfg.codebook_dim,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        for &r in &cfg.upsample_rates {
            col = col.max(t * 2 * r * (c / 2));
            (t, c) = (t * r, c / 2);
            act = act.max(t * c);
            col = col.max(t * c * KERNEL);
        }
        let mut ids = gpu.alloc::<i32>(frames * (GROUPS + 1))?;
        gpu.write(&mut ids, frames * GROUPS, &(0..frames as i32).collect::<Vec<_>>())?;
        Ok(Self {
            frames,
            ids,
            a: gpu.alloc(frames * act)?,
            b: gpu.alloc(frames * act)?,
            c: gpu.alloc(frames * act)?,
            residual: gpu.alloc(frames * cfg.hidden_size)?,
            col: gpu.alloc(frames * col)?,
            wav: gpu.alloc(frames * samples_per_frame)?,
        })
    }
}
