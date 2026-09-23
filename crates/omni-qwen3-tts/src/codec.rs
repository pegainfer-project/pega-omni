//! The codec decoder, streamed: every call turns one new frame of each of up
//! to `seqs` streams into 1920 samples, with nothing recomputed.
//!
//! RVQ lookup → causal conv → sliding-window transformer → two ×2 upsamplers
//! (transposed conv + ConvNeXt) → four SnakeBeta/transposed-conv blocks
//! (×8, ×5, ×4, ×3) with dilated residual units → SnakeBeta, conv, clamp.
//! Every layer is causal, so what a frame needs from the past is small and
//! fixed: each conv's last `(k - 1) · dilation` input rows, each overlapping
//! transposed conv's last GEMM row, each attention layer's last 72 frames of K
//! and V. A sequence's slot of the per-sequence state holds exactly that, and a
//! call reads and advances it (`kernels/codec.cu` has the layout). The output
//! is the whole-utterance decode's, frame by frame.
//!
//! [`build`] emits the decoder's calls from the checkpoint's config, the tail
//! of the model's `first` and `decode` programs. Activations are `[rows, C]`;
//! every dense conv is `im2col` and a GEMM, every transposed conv a GEMM and an
//! overlap-add; a residual branch's last GEMM accumulates into its stream.
//! Weight layout transforms and scale folding happen on the host at load
//! (LayerScale into the projections it scales, ConvNeXt's gamma into its last
//! linear, a bias into the linear op it feeds), then reach kern as named
//! tensors.

use anyhow::Result;
use anyhow::ensure;
use serde_json::json;

use crate::config;
use crate::manifest::Gen;
use crate::manifest::f32a;
use crate::manifest::i32a;
use crate::manifest::inb;
use crate::manifest::inf;
use crate::manifest::ini;
use crate::manifest::io;
use crate::manifest::outb;
use crate::manifest::per_seq;
use crate::manifest::state_io;
use crate::manifest::stride;
use crate::weights::File;
use crate::weights::Host;
use crate::weights::concat_rows;
use crate::weights::conv_taps;
use crate::weights::scale_rows;
use crate::weights::transposed_taps;

const GROUPS: usize = 16;
const KERNEL: usize = 7;

impl Gen {
    fn snake(&mut self, file: &File, prefix: &str, c: usize) -> Result<(String, String)> {
        let alpha = file.expect(&format!("{prefix}.alpha"), &[c])?;
        let beta = file.expect(&format!("{prefix}.beta"), &[c])?;
        Ok((
            self.weight_f32(&format!("{prefix}.a"), alpha.data.iter().map(|x| x.exp()).collect()),
            self.weight_f32(&format!("{prefix}.inv_b"), beta.data.iter().map(|x| 1.0 / (x.exp() + 1e-9)).collect()),
        ))
    }

    /// `out = conv(act(x + bias))` for a causal conv of `k` taps dilated by
    /// `d`, `act` SnakeBeta when given; the conv's own bias is left to its
    /// consumer. `act(x + bias)` of the last `(k - 1)·d` rows is its history.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        label: &str,
        (x, bias, act): (&str, &str, Option<&(String, String)>),
        (w, out): (&str, &'static str),
        t: usize,
        (cin, cout): (usize, usize),
        (k, d): (usize, usize),
    ) {
        let h = (k - 1) * d;
        let history = self.region(2 * h * cin * 2);
        self.need("col", t * k * cin);
        let mut args = vec![inb(x), inb(bias)];
        let entry = match act {
            Some((a, inv_b)) => {
                args.extend([inf(a), inf(inv_b)]);
                "codec_im2col_snake"
            }
            None => "codec_im2col",
        };
        args.extend([
            state_io(history),
            ini("pos"),
            ini("lines"),
            outb("col"),
            i32a(t),
            i32a(cin),
            i32a(k),
            i32a(d),
            stride(),
        ]);
        self.each8(&format!("{label}.im2col"), entry, (h + t) * cin, args);
        self.gemm(&format!("{label}.gemm"), out, "col", w, t, (cout, k * cin));
    }
}

/// `w[n, k] · v[k]`.
fn matvec(w: &[f32], v: &[f32]) -> Vec<f32> {
    w.chunks(v.len()).map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

fn plus(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// A `[rows, cols]` matrix as `[cols, rows]`.
fn transpose(w: &[f32], rows: usize) -> Vec<f32> {
    let cols = w.len() / rows;
    (0..cols).flat_map(|j| (0..rows).map(move |r| w[r * cols + j])).collect()
}

/// Emits the decoder's calls into `g`: `codes` [seqs, 16] in, `wav` [seqs, spf]
/// out, each sequence at frame `pos` of the stream its `lines` slot holds.
pub fn build(g: &mut Gen, file: &File, cfg: &config::Codec, spf: usize) -> Result<()> {
    let (dim, cb, latent, hidden) = (cfg.codebook_dim, cfg.codebook_size, cfg.latent_dim, cfg.hidden_size);
    let half = dim / 2;
    ensure!(cfg.num_quantizers == GROUPS, "{} quantizers, the decoder is built for {GROUPS}", cfg.num_quantizers);
    ensure!(cfg.head_dim == 64, "codec head_dim {} unsupported (attention is built for 64)", cfg.head_dim);
    ensure!(cfg.sliding_window == 72, "codec window {} unsupported (the KV ring is built for 72)", cfg.sliding_window);
    ensure!(
        cfg.upsampling_ratios.iter().chain(&cfg.upsample_rates).product::<usize>() == spf,
        "the upsampling rates do not multiply to {spf} samples per frame"
    );
    ensure!(
        hidden.is_multiple_of(256) && hidden <= 1024,
        "codec hidden size {hidden} unsupported (the norms need 256 | h ≤ 1024)"
    );
    ensure!(latent.is_multiple_of(8) && latent / 8 <= 1024, "codec latent dim {latent} unsupported");
    let out_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
    ensure!(
        out_dim.is_multiple_of(32) && out_dim <= 128,
        "codec output conv width {out_dim} unsupported (32 | c ≤ 128)"
    );
    g.module = "codec";

    // RVQ: [first | Σ rest] → one projection.
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
        .chain((0..GROUPS - 1).map(|i| codebook(&format!("decoder.quantizer.rvq_rest.vq.layers.{i}"))))
        .collect::<Result<Vec<_>>>()?
        .concat();
    let books = g.weight("rvq.books", &[GROUPS * cb, half], &books);
    let proj = |p: &str| file.expect(&format!("decoder.quantizer.{p}.output_proj.weight"), &[dim, half, 1]);
    let (first, rest) = (proj("rvq_first")?, proj("rvq_rest")?);
    let rvq_out: Vec<f32> = (0..dim)
        .flat_map(|r| first.data[r * half..(r + 1) * half].iter().chain(&rest.data[r * half..(r + 1) * half]))
        .copied()
        .collect();
    let rvq_out = g.weight("rvq.out", &[dim, dim], &rvq_out);
    g.launch(
        "rvq",
        "codec_rvq",
        [json!("seqs"), json!(1), json!(1)],
        (half / 8) as u32,
        vec![ini("codes"), inb(&books), outb("col"), i32a(half), i32a(cb)],
    );
    g.need("col", dim);
    g.gemm("rvq.proj", "a", "col", &rvq_out, 1, (dim, dim));

    let conv_w = |g: &mut Gen, prefix: &str, name: &str, shape: [usize; 3]| -> Result<(String, Vec<f32>)> {
        let w = file.expect(&format!("{prefix}.weight"), &shape)?;
        let b = file.expect(&format!("{prefix}.bias"), &shape[..1])?;
        Ok((g.weight(&format!("{name}.w"), &[shape[0], shape[2] * shape[1]], &conv_taps(&w)), b.data))
    };
    let zeros = g.weight("zeros", &[dim.max(latent)], &vec![0.0; dim.max(latent)]);
    let (pre, pre_b) = conv_w(g, "decoder.pre_conv.conv", "pre_conv", [latent, dim, 3])?;
    g.conv("pre_conv", ("a", &zeros, None), (&pre, "b"), 1, (dim, latent), (3, 1));

    // Transformer over the frames, residual stream in `res`; pre_conv's bias
    // goes through input_proj.
    let pt = "decoder.pre_transformer";
    let linear = |p: &str, shape: [usize; 2]| -> Result<(Host, Host)> {
        Ok((file.expect(&format!("{p}.weight"), &shape)?, file.expect(&format!("{p}.bias"), &shape[..1])?))
    };
    let (in_w, in_b) = linear(&format!("{pt}.input_proj"), [hidden, latent])?;
    let in_b = g.weight("input_proj.b", &[hidden], &plus(&in_b.data, &matvec(&in_w.data, &pre_b)));
    let in_w = g.weight("input_proj.w", &[hidden, latent], &in_w.data);
    g.gemm("input_proj", "res", "b", &in_w, 1, (hidden, latent));
    let (heads, hd, inter) = (cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size);
    let qd = heads * hd;
    let norm_w = |g: &mut Gen, name: &str, src: &str| -> Result<String> {
        Ok(g.weight(name, &[hidden], &file.expect(src, &[hidden])?.data))
    };
    let ln = norm_w(g, "l0.ln1", &format!("{pt}.layers.0.input_layernorm.weight"))?;
    let eps = cfg.rms_norm_eps;
    let norm_grid = [json!({"ceil_div": ["seqs", 4]}), json!(1), json!(1)];
    g.launch(
        "l0.norm",
        "codec_bias_rms_norm",
        norm_grid.clone(),
        128,
        vec![io("res"), inb(&in_b), inb(&ln), outb("x"), i32a(hidden), f32a(eps), ("i32", json!({"var": "seqs"}))],
    );
    g.need("x", hidden);
    g.need("a", qd.max(inter));
    for i in 0..cfg.num_hidden_layers {
        let w = |n: &str, shape: &[usize]| file.expect(&format!("{pt}.layers.{i}.{n}"), shape);
        let attn_scale = w("self_attn_layer_scale.scale", &[hidden])?;
        let mlp_scale = w("mlp_layer_scale.scale", &[hidden])?;
        let qkv = concat_rows(&[
            w("self_attn.q_proj.weight", &[qd, hidden])?,
            w("self_attn.k_proj.weight", &[qd, hidden])?,
            w("self_attn.v_proj.weight", &[qd, hidden])?,
        ]);
        let qkv = g.weight(&format!("l{i}.qkv"), &[3 * qd, hidden], &qkv);
        let o = g.weight(
            &format!("l{i}.o"),
            &[hidden, qd],
            &scale_rows(&w("self_attn.o_proj.weight", &[hidden, qd])?.data, &attn_scale.data),
        );
        let ln2 = norm_w(g, &format!("l{i}.ln2"), &format!("{pt}.layers.{i}.post_attention_layernorm.weight"))?;
        let gate_up =
            concat_rows(&[w("mlp.gate_proj.weight", &[inter, hidden])?, w("mlp.up_proj.weight", &[inter, hidden])?]);
        let gate_up = g.weight(&format!("l{i}.gate_up"), &[2 * inter, hidden], &gate_up);
        let down = g.weight(
            &format!("l{i}.down"),
            &[hidden, inter],
            &scale_rows(&w("mlp.down_proj.weight", &[hidden, inter])?.data, &mlp_scale.data),
        );
        let next = if i + 1 < cfg.num_hidden_layers {
            norm_w(g, &format!("l{}.ln1", i + 1), &format!("{pt}.layers.{}.input_layernorm.weight", i + 1))?
        } else {
            norm_w(g, "norm", &format!("{pt}.norm.weight"))?
        };
        let add_norm = |g: &mut Gen, label: String, w: &str| {
            g.launch(
                &label,
                "codec_add_rms_norm",
                norm_grid.clone(),
                128,
                vec![io("x"), io("res"), inb(w), i32a(hidden), f32a(eps), ("i32", json!({"var": "seqs"}))],
            );
        };

        let kv = g.region(2 * 72 * qd * 2);
        g.gemm(&format!("l{i}.qkv"), "col", "x", &qkv, 1, (3 * qd, hidden));
        g.launch(
            &format!("l{i}.attn"),
            "codec_attention",
            [json!("seqs"), json!(heads), json!(1)],
            128,
            vec![
                inb("col"),
                ini("pos"),
                ini("lines"),
                state_io(kv),
                outb("a"),
                stride(),
                i32a(heads),
                f32a(cfg.rope_theta),
            ],
        );
        g.gemm(&format!("l{i}.o"), "x", "a", &o, 1, (hidden, qd));
        add_norm(g, format!("l{i}.post_attn_norm"), &ln2);
        g.gemm(&format!("l{i}.gate_up"), "col", "x", &gate_up, 1, (2 * inter, hidden));
        g.each8(&format!("l{i}.silu_mul"), "codec_silu_mul", inter, vec![inb("col"), outb("a"), i32a(inter)]);
        g.gemm(&format!("l{i}.down"), "x", "a", &down, 1, (hidden, inter));
        add_norm(g, format!("l{i}.next_norm"), &next);
    }
    let (out_w, out_b) = linear(&format!("{pt}.output_proj"), [latent, hidden])?;
    let out_w = g.weight("output_proj.w", &[latent, hidden], &out_w.data);
    g.gemm("output_proj", "a", "x", &out_w, 1, (latent, hidden));

    // Upsamplers: transposed conv (kernel = stride, no overlap: its GEMM rows
    // are the output rows), then ConvNeXt, whose residual accumulates into its
    // input. `cur` holds the stage input and `pending` the bias it still
    // lacks, which the transposed conv folds into its own (per tap).
    let (mut cur, mut tmp) = ("a", "b");
    let mut pending = out_b.data;
    let mut t = 1;
    for (i, &r) in cfg.upsampling_ratios.iter().enumerate() {
        let p = format!("decoder.upsample.{i}");
        let up = transposed_taps(&file.expect(&format!("{p}.0.conv.weight"), &[latent, latent, r])?);
        let up_b = file.expect(&format!("{p}.0.conv.bias"), &[latent])?;
        let zb: Vec<f32> = matvec(&up, &pending).chunks(latent).flat_map(|tap| plus(tap, &up_b.data)).collect();
        let up_w = g.weight(&format!("up{i}.w"), &[r * latent, latent], &up);
        let zb = g.weight(&format!("up{i}.b"), &[r, latent], &zb);
        let dw = file.expect(&format!("{p}.1.dwconv.conv.weight"), &[latent, 1, KERNEL])?;
        let dw_w = g.weight(&format!("up{i}.dw.w"), &[KERNEL, latent], &transpose(&dw.data, latent));
        let dw_b = g.weight(
            &format!("up{i}.dw.b"),
            &[latent],
            &file.expect(&format!("{p}.1.dwconv.conv.bias"), &[latent])?.data,
        );
        let ln_w =
            g.weight(&format!("up{i}.ln.w"), &[latent], &file.expect(&format!("{p}.1.norm.weight"), &[latent])?.data);
        let ln_b =
            g.weight(&format!("up{i}.ln.b"), &[latent], &file.expect(&format!("{p}.1.norm.bias"), &[latent])?.data);
        let (pw1_w, pw1_b) = linear(&format!("{p}.1.pwconv1"), [4 * latent, latent])?;
        let pw1_w = g.weight(&format!("up{i}.pw1.w"), &[4 * latent, latent], &pw1_w.data);
        let pw1_b = g.weight(&format!("up{i}.pw1.b"), &[4 * latent], &pw1_b.data);
        let gamma = file.expect(&format!("{p}.1.gamma"), &[latent])?;
        let pw2: Host = file.expect(&format!("{p}.1.pwconv2.weight"), &[latent, 4 * latent])?;
        let pw2_b = file.expect(&format!("{p}.1.pwconv2.bias"), &[latent])?;
        let pw2_w = g.weight(&format!("up{i}.pw2.w"), &[latent, 4 * latent], &scale_rows(&pw2.data, &gamma.data));

        g.gemm(&format!("up{i}.gemm"), "col", cur, &up_w, t, (r * latent, latent));
        t *= r;
        let history = g.region(2 * (KERNEL - 1) * latent * 2);
        g.launch(
            &format!("up{i}.dwconv_ln"),
            "codec_dwconv_ln",
            [per_seq(t), json!(1), json!(1)],
            (latent / 8) as u32,
            vec![
                inb("col"),
                inb(&zb),
                inb(&dw_w),
                inb(&dw_b),
                inb(&ln_w),
                inb(&ln_b),
                state_io(history),
                ini("pos"),
                ini("lines"),
                outb(tmp),
                outb("x"),
                i32a(t),
                i32a(r),
                i32a(latent),
                stride(),
                f32a(1e-6),
            ],
        );
        g.gemm(&format!("up{i}.pw1"), "col", "x", &pw1_w, t, (4 * latent, latent));
        g.each8(
            &format!("up{i}.pw1.bias"),
            "codec_bias_gelu",
            t * 4 * latent,
            vec![io("col"), inb(&pw1_b), i32a(4 * latent)],
        );
        g.gemm_acc(&format!("up{i}.pw2"), tmp, "col", &pw2_w, t, (latent, 4 * latent));
        pending = scale_rows(&pw2_b.data, &gamma.data);
        g.need("x", t * latent);
        (cur, tmp) = (tmp, cur);
    }

    let pending_w = g.weight("conv_in.in_b", &[latent], &pending);
    let (conv_in, conv_in_b) = conv_w(g, "decoder.decoder.0.conv", "conv_in", [cfg.decoder_dim, latent, KERNEL])?;
    g.conv("conv_in", (cur, &pending_w, None), (&conv_in, tmp), t, (latent, cfg.decoder_dim), (KERNEL, 1));
    (cur, tmp) = (tmp, cur);
    let mut pending = conv_in_b;

    // Decoder blocks: SnakeBeta, transposed conv (overlap-add into `cur`),
    // then residual units whose second conv accumulates into `cur`, its bias
    // carried to the next consumer.
    for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
        let p = format!("decoder.decoder.{}.block", i + 1);
        let (cin, cout) = (cfg.decoder_dim >> i, cfg.decoder_dim >> (i + 1));
        let snake = g.snake(file, &format!("{p}.0"), cin)?;
        let in_b = g.weight(&format!("b{i}.in_b"), &[cin], &pending);
        let up = file.expect(&format!("{p}.1.conv.weight"), &[cin, cout, 2 * rate])?;
        let up_w = g.weight(&format!("b{i}.up.w"), &[2 * rate * cout, cin], &transposed_taps(&up));
        let up_b = g.weight(&format!("b{i}.up.b"), &[cout], &file.expect(&format!("{p}.1.conv.bias"), &[cout])?.data);

        g.each8(
            &format!("b{i}.snake"),
            "codec_bias_snake",
            t * cin,
            vec![inb(cur), inb(&in_b), inf(&snake.0), inf(&snake.1), outb(tmp), i32a(cin)],
        );
        g.gemm(&format!("b{i}.up"), "col", tmp, &up_w, t, (2 * rate * cout, cin));
        let prev = g.region(rate * cout * 2);
        g.each8(
            &format!("b{i}.col2im"),
            "codec_col2im",
            t * rate * cout,
            vec![
                inb("col"),
                inb(&up_b),
                state_io(prev),
                ini("lines"),
                outb(cur),
                i32a(t),
                i32a(rate),
                i32a(cout),
                stride(),
            ],
        );
        g.need(tmp, t * cin);
        t *= rate;
        g.need(cur, t * cout);
        g.need("x", t * cout);
        pending = vec![0.0; cout];
        for (u, dilation) in [1, 3, 9].into_iter().enumerate() {
            let q = format!("{p}.{}", u + 2);
            let label = format!("b{i}.u{u}");
            let s1 = g.snake(file, &format!("{q}.act1"), cout)?;
            let (c1, c1_b) = conv_w(g, &format!("{q}.conv1.conv"), &format!("{label}.conv1"), [cout, cout, KERNEL])?;
            let c1_b = g.weight(&format!("{label}.conv1.b"), &[cout], &c1_b);
            let s2 = g.snake(file, &format!("{q}.act2"), cout)?;
            let (c2, c2_b) = conv_w(g, &format!("{q}.conv2.conv"), &format!("{label}.conv2"), [cout, cout, 1])?;
            let in_b = g.weight(&format!("{label}.in_b"), &[cout], &pending);

            g.conv(&label, (cur, &in_b, Some(&s1)), (&c1, tmp), t, (cout, cout), (KERNEL, dilation));
            g.each8(
                &format!("{label}.snake2"),
                "codec_bias_snake",
                t * cout,
                vec![inb(tmp), inb(&c1_b), inf(&s2.0), inf(&s2.1), outb("x"), i32a(cout)],
            );
            g.gemm_acc(&format!("{label}.conv2"), cur, "x", &c2, t, (cout, cout));
            pending = plus(&pending, &c2_b);
        }
    }

    let n = cfg.upsample_rates.len() + 1;
    let snake_out = g.snake(file, &format!("decoder.decoder.{n}"), out_dim)?;
    let in_b = g.weight("conv_out.in_b", &[out_dim], &pending);
    let (conv_out, conv_out_b) =
        conv_w(g, &format!("decoder.decoder.{}.conv", n + 1), "conv_out", [1, out_dim, KERNEL])?;
    let conv_out_b = g.weight("conv_out.b", &[1], &conv_out_b);
    ensure!(t.is_multiple_of(64), "the output conv needs whole 64-row tiles, got {t} rows per frame");
    let history = g.region(2 * (KERNEL - 1) * out_dim * 2);
    g.launch(
        "conv_out",
        "codec_conv_out",
        [json!(t.div_ceil(64)), json!("seqs"), json!(1)],
        256,
        vec![
            inb(cur),
            inb(&in_b),
            inf(&snake_out.0),
            inf(&snake_out.1),
            inb(&conv_out),
            inb(&conv_out_b),
            state_io(history),
            ini("pos"),
            ini("lines"),
            outb("wav"),
            i32a(t),
            i32a(out_dim),
            stride(),
        ],
    );
    debug_assert_eq!(t, spf);
    g.need("res", hidden);
    Ok(())
}
