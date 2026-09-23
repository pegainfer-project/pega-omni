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
//! overlap-add. Weight layout transforms and scale folding happen on the host
//! at load (LayerScale into the projections it scales, ConvNeXt's gamma into
//! its last linear), then reach kern as named tensors.

use anyhow::Result;
use anyhow::ensure;
use serde_json::json;

use crate::config;
use crate::manifest::Gen;
use crate::manifest::THREADS;
use crate::manifest::i32a;
use crate::manifest::inb;
use crate::manifest::inf;
use crate::manifest::ini;
use crate::manifest::io;
use crate::manifest::outb;
use crate::manifest::per_seq;
use crate::manifest::state_in;
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

    fn bias(&mut self, label: &str, x: &str, b: &str, t: usize, c: usize) {
        self.each(label, "codec_bias", t * c, vec![io(x), inb(b), i32a(c)]);
    }

    /// `out = conv(x) + b` for a causal conv of `k` taps, the input's history
    /// in a ring of its own.
    fn conv(
        &mut self,
        label: &str,
        (x, out): (&str, &'static str),
        w: &(String, String),
        t: usize,
        (cin, cout): (usize, usize),
        k: usize,
    ) {
        let h = k - 1;
        let ring = self.region(h * cin * 2);
        self.need("col", t * k * cin);
        self.each8(
            &format!("{label}.im2col"),
            "codec_im2col",
            t * k * cin,
            vec![
                inb(x),
                state_in(ring),
                ini("pos"),
                ini("lines"),
                outb("col"),
                i32a(t),
                i32a(cin),
                i32a(k),
                i32a(1),
                i32a(h),
                stride(),
            ],
        );
        self.ring_write(&format!("{label}.history"), x, ring, t, cin, h);
        self.gemm(&format!("{label}.gemm"), out, "col", &w.0, t, (cout, k * cin));
        self.bias(&format!("{label}.bias"), out, &w.1, t, cout);
    }

    fn ring_write(&mut self, label: &str, x: &str, ring: u64, t: usize, c: usize, h: usize) {
        let w = t.min(h);
        self.each(
            label,
            "codec_ring_write",
            w * c,
            vec![inb(x), state_io(ring), ini("pos"), ini("lines"), i32a(t), i32a(c), i32a(h), i32a(w), stride()],
        );
    }
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
        THREADS,
        vec![ini("codes"), inb(&books), outb("col"), i32a(half), i32a(cb)],
    );
    g.need("col", dim);
    g.gemm("rvq.proj", "a", "col", &rvq_out, 1, (dim, dim));

    let conv_w = |g: &mut Gen, prefix: &str, name: &str, shape: [usize; 3]| -> Result<(String, String)> {
        let w = file.expect(&format!("{prefix}.weight"), &shape)?;
        let b = file.expect(&format!("{prefix}.bias"), &shape[..1])?;
        Ok((
            g.weight(&format!("{name}.w"), &[shape[0], shape[2] * shape[1]], &conv_taps(&w)),
            g.weight(&format!("{name}.b"), &[shape[0]], &b.data),
        ))
    };
    let pre = conv_w(g, "decoder.pre_conv.conv", "pre_conv", [latent, dim, 3])?;
    g.conv("pre_conv", ("a", "b"), &pre, 1, (dim, latent), 3);

    // Transformer over the frames, residual stream in `res`.
    let pt = "decoder.pre_transformer";
    let linear = |g: &mut Gen, p: &str, name: &str, shape: [usize; 2]| -> Result<(String, String)> {
        Ok((
            g.weight(&format!("{name}.w"), &shape, &file.expect(&format!("{p}.weight"), &shape)?.data),
            g.weight(&format!("{name}.b"), &shape[..1], &file.expect(&format!("{p}.bias"), &shape[..1])?.data),
        ))
    };
    let input_proj = linear(g, &format!("{pt}.input_proj"), "input_proj", [hidden, latent])?;
    g.gemm("input_proj", "res", "b", &input_proj.0, 1, (hidden, latent));
    g.bias("input_proj.bias", "res", &input_proj.1, 1, hidden);
    let (heads, hd, inter) = (cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size);
    let qd = heads * hd;
    let norm_w = |g: &mut Gen, name: &str, src: &str| -> Result<String> {
        Ok(g.weight(name, &[hidden], &file.expect(src, &[hidden])?.data))
    };
    let ln = norm_w(g, "l0.ln1", &format!("{pt}.layers.0.input_layernorm.weight"))?;
    let eps = cfg.rms_norm_eps;
    g.launch(
        "l0.norm",
        "codec_rms_norm",
        [json!("seqs"), json!(1), json!(1)],
        THREADS,
        vec![inb("res"), inb(&ln), outb("x"), i32a(hidden), ("f32", json!({"f32": eps}))],
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

        let kv = g.region(2 * 72 * qd * 2);
        g.gemm(&format!("l{i}.qkv"), "col", "x", &qkv, 1, (3 * qd, hidden));
        g.launch(
            &format!("l{i}.rope"),
            "codec_rope_kv",
            [json!("seqs"), json!((3 * heads).div_ceil(8)), json!(1)],
            THREADS,
            vec![
                io("col"),
                ini("pos"),
                ini("lines"),
                state_io(kv),
                stride(),
                i32a(heads),
                ("f32", json!({"f32": cfg.rope_theta})),
            ],
        );
        g.launch(
            &format!("l{i}.attn"),
            "codec_ring_attention",
            [json!("seqs"), json!(heads.div_ceil(8)), json!(1)],
            THREADS,
            vec![inb("col"), ini("pos"), ini("lines"), state_in(kv), outb("a"), stride(), i32a(heads)],
        );
        g.gemm(&format!("l{i}.o"), "x", "a", &o, 1, (hidden, qd));
        g.launch(
            &format!("l{i}.post_attn_norm"),
            "codec_add_rms_norm",
            [json!("seqs"), json!(1), json!(1)],
            THREADS,
            vec![io("x"), io("res"), inb(&ln2), i32a(hidden), ("f32", json!({"f32": eps}))],
        );
        g.gemm(&format!("l{i}.gate_up"), "col", "x", &gate_up, 1, (2 * inter, hidden));
        g.each(&format!("l{i}.silu_mul"), "codec_silu_mul", inter, vec![inb("col"), outb("a"), i32a(inter)]);
        g.gemm(&format!("l{i}.down"), "x", "a", &down, 1, (hidden, inter));
        g.launch(
            &format!("l{i}.next_norm"),
            "codec_add_rms_norm",
            [json!("seqs"), json!(1), json!(1)],
            THREADS,
            vec![io("x"), io("res"), inb(&next), i32a(hidden), ("f32", json!({"f32": eps}))],
        );
    }
    let output_proj = linear(g, &format!("{pt}.output_proj"), "output_proj", [latent, hidden])?;
    g.gemm("output_proj", "a", "x", &output_proj.0, 1, (latent, hidden));
    g.bias("output_proj.bias", "a", &output_proj.1, 1, latent);

    // Upsamplers: transposed conv (kernel = stride, no overlap), then ConvNeXt
    // with its residual. `cur` holds the stage input, `tmp` is free.
    let (mut cur, mut tmp) = ("a", "b");
    let mut t = 1;
    for (i, &r) in cfg.upsampling_ratios.iter().enumerate() {
        let p = format!("decoder.upsample.{i}");
        let up = file.expect(&format!("{p}.0.conv.weight"), &[latent, latent, r])?;
        let up_w = g.weight(&format!("up{i}.w"), &[r * latent, latent], &transposed_taps(&up));
        let up_b = g.weight(&format!("up{i}.b"), &[latent], &file.expect(&format!("{p}.0.conv.bias"), &[latent])?.data);
        let dw_w = g.weight(
            &format!("up{i}.dw.w"),
            &[latent, KERNEL],
            &file.expect(&format!("{p}.1.dwconv.conv.weight"), &[latent, 1, KERNEL])?.data,
        );
        let dw_b = g.weight(
            &format!("up{i}.dw.b"),
            &[latent],
            &file.expect(&format!("{p}.1.dwconv.conv.bias"), &[latent])?.data,
        );
        let ln_w =
            g.weight(&format!("up{i}.ln.w"), &[latent], &file.expect(&format!("{p}.1.norm.weight"), &[latent])?.data);
        let ln_b =
            g.weight(&format!("up{i}.ln.b"), &[latent], &file.expect(&format!("{p}.1.norm.bias"), &[latent])?.data);
        let pw1 = linear(g, &format!("{p}.1.pwconv1"), &format!("up{i}.pw1"), [4 * latent, latent])?;
        let gamma = file.expect(&format!("{p}.1.gamma"), &[latent])?;
        let pw2: Host = file.expect(&format!("{p}.1.pwconv2.weight"), &[latent, 4 * latent])?;
        let pw2_b = file.expect(&format!("{p}.1.pwconv2.bias"), &[latent])?;
        let pw2_w = g.weight(&format!("up{i}.pw2.w"), &[latent, 4 * latent], &scale_rows(&pw2.data, &gamma.data));
        let pw2_b = g.weight(&format!("up{i}.pw2.b"), &[latent], &scale_rows(&pw2_b.data, &gamma.data));

        g.gemm(&format!("up{i}.gemm"), "col", cur, &up_w, t, (r * latent, latent));
        t *= r;
        g.each(
            &format!("up{i}.unfold"),
            "codec_unfold",
            t * latent,
            vec![inb("col"), inb(&up_b), outb(tmp), i32a(latent)],
        );
        let h = KERNEL - 1;
        let ring = g.region(h * latent * 2);
        g.launch(
            &format!("up{i}.dwconv_ln"),
            "codec_dwconv_ln",
            [per_seq(t), json!(1), json!(1)],
            THREADS,
            vec![
                inb(tmp),
                inb(&dw_w),
                inb(&dw_b),
                inb(&ln_w),
                inb(&ln_b),
                state_in(ring),
                ini("pos"),
                ini("lines"),
                outb("x"),
                i32a(t),
                i32a(latent),
                i32a(KERNEL),
                stride(),
                ("f32", json!({"f32": 1e-6f32})),
            ],
        );
        g.ring_write(&format!("up{i}.history"), tmp, ring, t, latent, h);
        g.gemm(&format!("up{i}.pw1"), "col", "x", &pw1.0, t, (4 * latent, latent));
        g.each(
            &format!("up{i}.pw1.bias"),
            "codec_bias_gelu",
            t * 4 * latent,
            vec![io("col"), inb(&pw1.1), i32a(4 * latent)],
        );
        g.gemm(&format!("up{i}.pw2"), "x", "col", &pw2_w, t, (latent, 4 * latent));
        g.each(
            &format!("up{i}.residual"),
            "codec_bias_residual",
            t * latent,
            vec![inb("x"), inb(&pw2_b), io(tmp), i32a(latent)],
        );
        g.need(tmp, t * latent);
        g.need("x", t * latent);
        (cur, tmp) = (tmp, cur);
    }

    let conv_in = conv_w(g, "decoder.decoder.0.conv", "conv_in", [cfg.decoder_dim, latent, KERNEL])?;
    g.conv("conv_in", (cur, tmp), &conv_in, t, (latent, cfg.decoder_dim), KERNEL);
    (cur, tmp) = (tmp, cur);

    for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
        let p = format!("decoder.decoder.{}.block", i + 1);
        let (cin, cout) = (cfg.decoder_dim >> i, cfg.decoder_dim >> (i + 1));
        let snake = g.snake(file, &format!("{p}.0"), cin)?;
        let up = file.expect(&format!("{p}.1.conv.weight"), &[cin, cout, 2 * rate])?;
        let up_w = g.weight(&format!("b{i}.up.w"), &[2 * rate * cout, cin], &transposed_taps(&up));
        let up_b = g.weight(&format!("b{i}.up.b"), &[cout], &file.expect(&format!("{p}.1.conv.bias"), &[cout])?.data);

        g.each8(
            &format!("b{i}.snake"),
            "codec_snake",
            t * cin,
            vec![inb(cur), inf(&snake.0), inf(&snake.1), outb(tmp), i32a(cin)],
        );
        g.gemm(&format!("b{i}.up"), "col", tmp, &up_w, t, (2 * rate * cout, cin));
        let prev = g.region(rate * cout * 2);
        g.each(
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
        for (u, dilation) in [1, 3, 9].into_iter().enumerate() {
            let q = format!("{p}.{}", u + 2);
            let s1 = g.snake(file, &format!("{q}.act1"), cout)?;
            let c1 = conv_w(g, &format!("{q}.conv1.conv"), &format!("b{i}.u{u}.conv1"), [cout, cout, KERNEL])?;
            let s2 = g.snake(file, &format!("{q}.act2"), cout)?;
            let c2 = conv_w(g, &format!("{q}.conv2.conv"), &format!("b{i}.u{u}.conv2"), [cout, cout, 1])?;
            let label = format!("b{i}.u{u}");
            let h = (KERNEL - 1) * dilation;
            let ring = g.region(h * cout * 2);
            g.need("col", t * KERNEL * cout);
            g.each8(
                &format!("{label}.snake1"),
                "codec_snake",
                t * cout,
                vec![inb(cur), inf(&s1.0), inf(&s1.1), outb("x"), i32a(cout)],
            );
            g.each8(
                &format!("{label}.im2col"),
                "codec_im2col",
                t * KERNEL * cout,
                vec![
                    inb("x"),
                    state_in(ring),
                    ini("pos"),
                    ini("lines"),
                    outb("col"),
                    i32a(t),
                    i32a(cout),
                    i32a(KERNEL),
                    i32a(dilation),
                    i32a(h),
                    stride(),
                ],
            );
            g.ring_write(&format!("{label}.history"), "x", ring, t, cout, h);
            g.gemm(&format!("{label}.conv1"), tmp, "col", &c1.0, t, (cout, KERNEL * cout));
            g.each(
                &format!("{label}.snake2"),
                "codec_bias_snake",
                t * cout,
                vec![inb(tmp), inb(&c1.1), inf(&s2.0), inf(&s2.1), outb("x"), i32a(cout)],
            );
            g.gemm(&format!("{label}.conv2"), tmp, "x", &c2.0, t, (cout, cout));
            g.each(
                &format!("{label}.residual"),
                "codec_bias_residual",
                t * cout,
                vec![inb(tmp), inb(&c2.1), io(cur), i32a(cout)],
            );
        }
    }

    let n = cfg.upsample_rates.len() + 1;
    let out_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
    let snake_out = g.snake(file, &format!("decoder.decoder.{n}"), out_dim)?;
    let conv_out = conv_w(g, &format!("decoder.decoder.{}.conv", n + 1), "conv_out", [1, out_dim, KERNEL])?;
    let h = KERNEL - 1;
    let ring = g.region(h * out_dim * 2);
    g.launch(
        "conv_out",
        "codec_conv_out",
        [json!({"ceil_div": [per_seq(t), 8]}), json!(1), json!(1)],
        THREADS,
        vec![
            inb(cur),
            inf(&snake_out.0),
            inf(&snake_out.1),
            inb(&conv_out.0),
            inb(&conv_out.1),
            state_in(ring),
            ini("pos"),
            ini("lines"),
            outb("wav"),
            i32a(t),
            i32a(out_dim),
            i32a(KERNEL),
            stride(),
            ("i32", json!({"expr": per_seq(t)})),
        ],
    );
    g.ring_write("conv_out.history", cur, ring, t, out_dim, h);
    debug_assert_eq!(t, spf);

    g.need("res", hidden);
    Ok(())
}
