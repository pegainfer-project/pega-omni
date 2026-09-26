//! Mimi, streamed, as manifest calls: [`encode`] turns each session's
//! 1920-sample caller frame into 8 codes, [`decode`] turns the agent's 8
//! codes into 1920 samples, both one frame per call with nothing recomputed.
//!
//! Encoder: SEANet (a conv, then four stages of a residual block and a
//! strided conv, ×4 ×5 ×6 ×8 down to 25 Hz, then a conv) → an 8-layer
//! transformer over the last 250 rows → a stride-2 conv to 12.5 Hz → split
//! residual quantization (a semantic codebook on one projection, seven
//! acoustic ones residually on another). The decoder mirrors it: codebook
//! lookup and projection → a depthwise stride-2 transposed conv → a
//! transformer → SEANet with transposed convs ×8 ×6 ×5 ×4 up to 24 kHz.
//!
//! Every layer is causal, so what a frame needs from the past is small and
//! fixed; each layer takes its piece of the session's slot (`kernels/mimi.cu`
//! has the layout). Activations are `[rows, C]`; a conv is `im2col` and a
//! GEMM, a transposed conv a GEMM and an overlap-add. A conv's bias rides to
//! its consumer as a `pending` bias (added on load), so a residual branch's
//! last GEMM can accumulate straight into its stream. LayerScale is folded
//! into the projection it scales at load.

use anyhow::Result;
use omni_kern::Gen;
use omni_kern::buf_at;
use omni_kern::f32a;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::inf;
use omni_kern::ini;
use omni_kern::io;
use omni_kern::outb;
use omni_kern::per_seq;
use omni_kern::state_in;
use omni_kern::state_io;
use omni_kern::stride;
use omni_kern::weights::File;
use omni_kern::weights::Host;
use omni_kern::weights::conv_taps;
use omni_kern::weights::scale_rows;
use omni_kern::weights::transposed_taps;
use serde_json::json;

use crate::config::CODEBOOK_DIM;
use crate::config::CODEBOOKS;
use crate::config::FRAME;
use crate::config::MIMI_CONTEXT;
use crate::config::MIMI_DIM;
use crate::config::MIMI_FILTERS;
use crate::config::MIMI_HEADS;
use crate::config::MIMI_HIDDEN;
use crate::config::MIMI_LAYERS;
use crate::config::RATIOS;
use crate::helium::rope_coef;

const BOOK: usize = 2048;
const KERNEL: usize = 7;
const RES_KERNEL: usize = 3;
const LAST_KERNEL: usize = 3;
const LN_EPS: f32 = 1e-5;
/// Rows per frame between the SEANet and the 12.5 Hz latent.
const T_LATENT: usize = 2;

/// Where a stage's activations live and the bias they still lack.
struct Stream {
    cur: &'static str,
    tmp: &'static str,
    pending: Vec<f32>,
}

impl Stream {
    fn swap(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.tmp);
    }
}

fn weight_bias(file: &File, prefix: &str, shape: [usize; 3]) -> Result<(Host, Vec<f32>)> {
    Ok((file.expect(&format!("{prefix}.weight"), &shape)?, file.expect(&format!("{prefix}.bias"), &shape[..1])?.data))
}

fn plus(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `out = conv(act(x + pending))` for a causal conv of kernel `k`, stride `s`
/// over `t` rows of `cin`; returns the conv's bias for the consumer.
#[allow(clippy::too_many_arguments)]
fn conv(
    g: &mut Gen,
    label: &str,
    x: &Stream,
    (w, bias): (Host, Vec<f32>),
    out: &'static str,
    t: usize,
    (k, s): (usize, usize),
    elu: bool,
    replicate: bool,
) -> Vec<f32> {
    let (cout, cin) = (w.shape[0], w.shape[1]);
    let history = g.region(2 * (k - s) * cin * 2);
    let pending = g.weight(&format!("{label}.in_b"), &[cin], &x.pending);
    let taps = g.weight(&format!("{label}.w"), &[cout, k * cin], &conv_taps(&w));
    g.need("m.col", t / s * k * cin);
    g.each8(
        &format!("{label}.im2col"),
        "mimi_im2col",
        &json!("seqs"),
        (k - s + t) * cin,
        vec![
            inb(x.cur),
            inb(&pending),
            state_io(history),
            ini("frame"),
            ini("lines"),
            outb("m.col"),
            i32a(t),
            i32a(cin),
            i32a(k),
            i32a(s),
            i32a(elu as usize),
            i32a(replicate as usize),
            stride(),
        ],
    );
    g.gemm(&format!("{label}.gemm"), out, "m.col", &taps, t / s, (cout, k * cin));
    bias
}

/// A SEANet residual block on `x` (`t` rows): ELU, conv k3 to half the
/// channels, ELU, conv k1 back, accumulated into the stream.
fn residual(g: &mut Gen, file: &File, label: &str, prefix: &str, x: &mut Stream, t: usize, c: usize) -> Result<()> {
    let first = weight_bias(file, &format!("{prefix}.block.1.conv.conv"), [c / 2, c, RES_KERNEL])?;
    let (second, b2) = weight_bias(file, &format!("{prefix}.block.3.conv.conv"), [c, c / 2, 1])?;
    let b1 = conv(g, &format!("{label}.conv1"), x, first, "m.h", t, (RES_KERNEL, 1), true, false);
    let b1 = g.weight(&format!("{label}.conv1.b"), &[c / 2], &b1);
    g.each8(
        &format!("{label}.act"),
        "mimi_bias_act",
        &per_seq(t),
        c / 2,
        vec![inb("m.h"), inb(&b1), outb("m.h"), i32a(c / 2), i32a(1)],
    );
    let w2 = g.weight(&format!("{label}.conv2.w"), &[c, c / 2], &second.data);
    g.gemm_acc(&format!("{label}.conv2"), x.cur, "m.h", &w2, t, (c, c / 2));
    x.pending = plus(&x.pending, &b2);
    Ok(())
}

/// An 8-layer transformer over `x` (two rows a frame), residual stream in
/// `x.cur`, its pending bias added on the way in.
fn transformer(g: &mut Gen, file: &File, label: &str, prefix: &str, x: &mut Stream) -> Result<()> {
    let (d, h) = (MIMI_DIM, MIMI_HIDDEN);
    let rows = per_seq(T_LATENT);
    let ln_grid = [json!({"ceil_div": [rows.clone(), 4]}), json!(1), json!(1)];
    let rows_arg = ("i32", json!({"expr": rows.clone()}));
    let vec_w = |g: &mut Gen, name: &str, src: &str| -> Result<String> {
        Ok(g.weight(name, &[d], &file.expect(src, &[d])?.data))
    };
    let in_b = g.weight(&format!("{label}.in_b"), &[d], &x.pending);
    x.pending = vec![0.0; d];
    for (name, w) in [("m.xn", d), ("m.qkv", 3 * d), ("m.attn", d), ("m.ff", h)] {
        g.need(name, T_LATENT * w);
    }
    for i in 0..MIMI_LAYERS {
        let p = format!("{prefix}.transformer.layers.{i}");
        let at = |s: &str| format!("{label}.l{i}.{s}");
        let w = |n: &str, shape: &[usize]| file.expect(&format!("{p}.{n}"), shape);
        let norm = |g: &mut Gen, n: &str| -> Result<(String, String)> {
            Ok((
                vec_w(g, &at(&format!("{n}.w")), &format!("{p}.{n}.weight"))?,
                vec_w(g, &at(&format!("{n}.b")), &format!("{p}.{n}.bias"))?,
            ))
        };
        let (n1w, n1b) = norm(g, "norm1")?;
        let (n2w, n2b) = norm(g, "norm2")?;
        let qkv = g.weight(&at("qkv"), &[3 * d, d], &w("self_attn.in_proj_weight", &[3 * d, d])?.data);
        let out = g.weight(
            &at("out"),
            &[d, d],
            &scale_rows(&w("self_attn.out_proj.weight", &[d, d])?.data, &w("layer_scale_1.scale", &[d])?.data),
        );
        let lin1 = g.weight(&at("lin1"), &[h, d], &w("linear1.weight", &[h, d])?.data);
        let lin2 = g.weight(
            &at("lin2"),
            &[d, h],
            &scale_rows(&w("linear2.weight", &[d, h])?.data, &w("layer_scale_2.scale", &[d])?.data),
        );
        let ln = |g: &mut Gen, label: String, (nw, nb): (&str, &str), add: bool| {
            g.launch(
                &label,
                "mimi_layer_norm",
                ln_grid.clone(),
                128,
                vec![
                    io(x.cur),
                    inb(&in_b),
                    i32a(add as usize),
                    inb(nw),
                    inb(nb),
                    outb("m.xn"),
                    i32a(d),
                    f32a(LN_EPS),
                    rows_arg.clone(),
                ],
            );
        };
        ln(g, at("norm1"), (&n1w, &n1b), i == 0);
        let ring = g.region(MIMI_CONTEXT * 2 * d * 2);
        g.gemm(&at("qkv"), "m.qkv", "m.xn", &qkv, T_LATENT, (3 * d, d));
        g.launch(
            &at("rope"),
            "mimi_rope_kv",
            [rows.clone(), json!((3 * MIMI_HEADS).div_ceil(8)), json!(1)],
            256,
            vec![
                io("m.qkv"),
                ini("frame"),
                ini("lines"),
                state_io(ring),
                i32a(T_LATENT),
                i32a(MIMI_HEADS),
                f32a(rope_coef(d / MIMI_HEADS)),
                i32a(MIMI_CONTEXT),
                stride(),
            ],
        );
        g.launch(
            &at("attn"),
            "mimi_attend",
            [rows.clone(), json!(MIMI_HEADS), json!(1)],
            128,
            vec![
                inb("m.qkv"),
                ini("frame"),
                ini("lines"),
                state_in(ring),
                outb("m.attn"),
                i32a(T_LATENT),
                i32a(MIMI_HEADS),
                f32a(1.0 / ((d / MIMI_HEADS) as f32).sqrt()),
                i32a(MIMI_CONTEXT),
                stride(),
            ],
        );
        g.gemm_acc(&at("out"), x.cur, "m.attn", &out, T_LATENT, (d, d));
        ln(g, at("norm2"), (&n2w, &n2b), false);
        g.gemm(&at("lin1"), "m.ff", "m.xn", &lin1, T_LATENT, (h, d));
        g.each8(&at("gelu"), "mimi_gelu", &rows, h, vec![io("m.ff")]);
        g.gemm_acc(&at("lin2"), x.cur, "m.ff", &lin2, T_LATENT, (d, h));
    }
    Ok(())
}

/// `m.lat` rows of the 8 codebooks in use, each `[2048, 256]` f32, and their squared norms.
fn codebooks(file: &File) -> Result<(Vec<f32>, Vec<f32>)> {
    let book = |p: String| -> Result<Vec<f32>> {
        let sum = file.expect(&format!("{p}._codebook.embedding_sum"), &[BOOK, CODEBOOK_DIM])?;
        let usage = file.expect(&format!("{p}._codebook.cluster_usage"), &[BOOK])?;
        Ok(sum
            .data
            .chunks(CODEBOOK_DIM)
            .zip(&usage.data)
            .flat_map(|(row, &u)| row.iter().map(move |x| x / u.max(1e-5)))
            .collect())
    };
    let books = std::iter::once(book("quantizer.rvq_first.vq.layers.0".into()))
        .chain((0..CODEBOOKS - 1).map(|i| book(format!("quantizer.rvq_rest.vq.layers.{i}"))))
        .collect::<Result<Vec<_>>>()?
        .concat();
    let norms =
        books.chunks(CODEBOOK_DIM).map(|e| e.iter().map(|&x| x as f64 * x as f64).sum::<f64>() as f32).collect();
    Ok((books, norms))
}

/// `caller` (f32 PCM) in, `caller_latent` (the quantizer's input) and `caller_codes` out.
pub fn encode(g: &mut Gen, file: &File) -> Result<()> {
    let seqs = json!("seqs");
    let w0 = file.expect("encoder.model.0.conv.conv.weight", &[MIMI_FILTERS, 1, KERNEL])?;
    let b0 = file.expect("encoder.model.0.conv.conv.bias", &[MIMI_FILTERS])?;
    let (w0, b0) = (g.weight_f32("enc.conv0.w", w0.data), g.weight_f32("enc.conv0.b", b0.data));
    let history = g.region(2 * (KERNEL - 1) * 4);
    g.need("m.a", FRAME * MIMI_FILTERS);
    g.each8(
        "enc.conv0",
        "mimi_conv_in",
        &seqs,
        FRAME * MIMI_FILTERS,
        vec![
            inf("caller"),
            inf(&w0),
            inf(&b0),
            state_io(history),
            ini("frame"),
            ini("lines"),
            outb("m.a"),
            i32a(FRAME),
            i32a(MIMI_FILTERS),
            i32a(KERNEL),
            stride(),
        ],
    );
    let mut x = Stream { cur: "m.a", tmp: "m.b", pending: vec![0.0; MIMI_FILTERS] };
    let (mut t, mut c) = (FRAME, MIMI_FILTERS);
    for (i, &r) in RATIOS.iter().rev().enumerate() {
        let base = 1 + 3 * i;
        g.need("m.h", t * c / 2);
        residual(g, file, &format!("enc.s{i}.res"), &format!("encoder.model.{base}"), &mut x, t, c)?;
        let down = weight_bias(file, &format!("encoder.model.{}.conv.conv", base + 2), [2 * c, c, 2 * r])?;
        x.pending = conv(g, &format!("enc.s{i}.down"), &x, down, x.tmp, t, (2 * r, r), true, false);
        x.swap();
        t /= r;
        c *= 2;
        g.need(x.cur, t * c);
    }
    let last = weight_bias(file, "encoder.model.14.conv.conv", [MIMI_DIM, c, LAST_KERNEL])?;
    x.pending = conv(g, "enc.last", &x, last, x.tmp, t, (LAST_KERNEL, 1), true, false);
    x.swap();
    transformer(g, file, "enc.tr", "encoder_transformer", &mut x)?;
    let down = file.expect("downsample.conv.conv.conv.weight", &[MIMI_DIM, MIMI_DIM, 4])?;
    conv(g, "enc.downsample", &x, (down, vec![]), "caller_latent", T_LATENT, (4, 2), false, true);
    let proj = |p: &str| file.expect(&format!("quantizer.{p}.input_proj.weight"), &[CODEBOOK_DIM, MIMI_DIM, 1]);
    let proj = g.weight(
        "enc.q.proj",
        &[2 * CODEBOOK_DIM, MIMI_DIM],
        &[proj("rvq_first")?.data, proj("rvq_rest")?.data].concat(),
    );
    g.gemm("enc.q.proj", "m.q", "caller_latent", &proj, 1, (2 * CODEBOOK_DIM, MIMI_DIM));
    let (books, norms) = codebooks(file)?;
    let (books, norms) = (g.weight_f32("q.books", books), g.weight_f32("q.norms", norms));
    g.launch(
        "enc.quantize",
        "mimi_quantize",
        [seqs, json!(1), json!(1)],
        CODEBOOK_DIM as u32,
        vec![
            inb("m.q"),
            inf(&books),
            inf(&norms),
            ("out buffer<i32>", omni_kern::buf("caller_codes")),
            i32a(CODEBOOKS),
        ],
    );
    Ok(())
}

/// The agent codes (`emitted` columns 1..=8) in, `pcm` out.
pub fn decode(g: &mut Gen, file: &File) -> Result<()> {
    let seqs = json!("seqs");
    g.need("m.q", 2 * CODEBOOK_DIM);
    g.each8(
        "dec.dequantize",
        "mimi_dequantize",
        &seqs,
        2 * CODEBOOK_DIM,
        vec![
            ("in buffer<i32>", buf_at("emitted", 4)),
            i32a(1 + CODEBOOKS),
            inf("q.books"),
            outb("m.q"),
            i32a(CODEBOOKS),
        ],
    );
    let proj = |p: &str| file.expect(&format!("quantizer.{p}.output_proj.weight"), &[MIMI_DIM, CODEBOOK_DIM, 1]);
    let (first, rest) = (proj("rvq_first")?, proj("rvq_rest")?);
    let out: Vec<f32> = (0..MIMI_DIM)
        .flat_map(|r| {
            first.data[r * CODEBOOK_DIM..(r + 1) * CODEBOOK_DIM]
                .iter()
                .chain(&rest.data[r * CODEBOOK_DIM..(r + 1) * CODEBOOK_DIM])
        })
        .copied()
        .collect();
    let out = g.weight("dec.q.proj", &[MIMI_DIM, 2 * CODEBOOK_DIM], &out);
    g.gemm("dec.q.proj", "m.lat", "m.q", &out, 1, (MIMI_DIM, 2 * CODEBOOK_DIM));
    let up =
        g.weight_f32("dec.upsample.w", file.expect("upsample.convtr.convtr.convtr.weight", &[MIMI_DIM, 1, 4])?.data);
    let prev = g.region(MIMI_DIM * 2);
    g.launch(
        "dec.upsample",
        "mimi_upsample",
        [json!({"ceil_div": [{"mul": ["seqs", MIMI_DIM]}, 256]}), json!(1), json!(1)],
        256,
        vec![
            inb("m.lat"),
            inf(&up),
            state_io(prev),
            ini("lines"),
            outb("m.a"),
            i32a(MIMI_DIM),
            stride(),
            ("i32", json!({"expr": {"mul": ["seqs", MIMI_DIM]}})),
        ],
    );
    let mut x = Stream { cur: "m.a", tmp: "m.b", pending: vec![0.0; MIMI_DIM] };
    transformer(g, file, "dec.tr", "decoder_transformer", &mut x)?;
    let (mut t, mut c) = (T_LATENT, 2 * MIMI_DIM);
    let first = weight_bias(file, "decoder.model.0.conv.conv", [c, MIMI_DIM, KERNEL])?;
    x.pending = conv(g, "dec.first", &x, first, x.tmp, t, (KERNEL, 1), false, false);
    x.swap();
    for (i, &r) in RATIOS.iter().enumerate() {
        let base = 2 + 3 * i;
        let p = format!("decoder.model.{base}.convtr.convtr");
        let (w, b) = (
            file.expect(&format!("{p}.weight"), &[c, c / 2, 2 * r])?,
            file.expect(&format!("{p}.bias"), &[c / 2])?.data,
        );
        let in_b = g.weight(&format!("dec.s{i}.in_b"), &[c], &x.pending);
        g.each8(
            &format!("dec.s{i}.act"),
            "mimi_bias_act",
            &per_seq(t),
            c,
            vec![inb(x.cur), inb(&in_b), outb(x.tmp), i32a(c), i32a(1)],
        );
        let taps = g.weight(&format!("dec.s{i}.up.w"), &[2 * r * c / 2, c], &transposed_taps(&w));
        let b = g.weight(&format!("dec.s{i}.up.b"), &[c / 2], &b);
        g.need(x.tmp, t * c);
        g.gemm(&format!("dec.s{i}.up"), "m.col", x.tmp, &taps, t, (2 * r * c / 2, c));
        let partial = g.region(r * c / 2 * 2);
        g.each8(
            &format!("dec.s{i}.col2im"),
            "mimi_col2im",
            &per_seq(t),
            r * c / 2,
            vec![
                inb("m.col"),
                inb(&b),
                state_io(partial),
                ini("lines"),
                outb(x.cur),
                i32a(t),
                i32a(r),
                i32a(c / 2),
                stride(),
            ],
        );
        t *= r;
        c /= 2;
        g.need(x.cur, t * c);
        g.need("m.h", t * c / 2);
        x.pending = vec![0.0; c];
        residual(g, file, &format!("dec.s{i}.res"), &format!("decoder.model.{}", base + 1), &mut x, t, c)?;
    }
    let w = file.expect("decoder.model.14.conv.conv.weight", &[1, c, LAST_KERNEL])?;
    let w = g.weight_f32("dec.last.w", conv_taps(&w));
    let b = g.weight_f32("dec.last.b", file.expect("decoder.model.14.conv.conv.bias", &[1])?.data);
    let in_b = g.weight("dec.last.in_b", &[c], &x.pending);
    let history = g.region(2 * (LAST_KERNEL - 1) * c * 2);
    g.launch(
        "dec.last",
        "mimi_conv_out",
        [json!({"ceil_div": [{"mul": ["seqs", t]}, 256]}), json!(1), json!(1)],
        256,
        vec![
            inb(x.cur),
            inb(&in_b),
            inf(&w),
            inf(&b),
            state_io(history),
            ini("frame"),
            ini("lines"),
            ("out buffer<f32>", omni_kern::buf("pcm")),
            i32a(t),
            i32a(c),
            i32a(LAST_KERNEL),
            stride(),
            ("i32", json!({"expr": {"mul": ["seqs", t]}})),
        ],
    );
    Ok(())
}
