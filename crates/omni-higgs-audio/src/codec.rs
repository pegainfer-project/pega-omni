//! The codec decoder as manifest calls: a window of frames in, PCM out.
//!
//! Quantizer lookup (folded with `fc2`, see [`crate::fold`]) → conv → five blocks of
//! Snake / transposed conv (×8, ×5, ×4, ×2, ×3) and three dilated residual units → Snake, conv.
//! Activations are `[rows, C]`; every dense conv is `im2col` and a GEMM, every transposed conv a
//! GEMM and an overlap-add. The weights stay in the checkpoint's bf16 and activations are bf16
//! between kernels, with f32 accumulation inside the GEMMs and kernels.
//!
//! Row counts are multiples of the `frames` var, so one program serves any window up to
//! [`MAX_FRAMES`].

use anyhow::Result;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;

use crate::config::CODEBOOKS;
use crate::config::SAMPLES_PER_FRAME;
use crate::fold::Fc2;
use crate::fold::Quantizer;
use crate::fold::fold;
use omni_kern::Gen;
use omni_kern::THREADS;
use omni_kern::buf;
use omni_kern::count;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::inf;
use omni_kern::ini;
use omni_kern::io;
use omni_kern::outb;
use omni_kern::weights::File;
use omni_kern::weights::conv_taps;
use omni_kern::weights::transposed_taps;

/// The longest window one call decodes.
pub const MAX_FRAMES: usize = 128;
/// Codes per codebook the codec reads (the model's last two codes are not codec codes).
pub const CODES: usize = 1024;
const PREFIX: &str = "tied.embedding.modality_embeddings.0.model.";
const WIDTH: usize = 64;
const LATENT: usize = 1024;
const ACOUSTIC: usize = 256;
const TAPS: usize = 7;
const STRIDES: [usize; 5] = [8, 5, 4, 2, 3];
const DILATIONS: [usize; 3] = [1, 3, 9];
const OUT_CHANNELS: usize = LATENT >> STRIDES.len();

/// Workspaces: the stream, the residual unit's branch, and the im2col / transposed-conv columns.
const STREAM: &str = "codec.a";
const BRANCH: &str = "codec.b";
const COLUMNS: &str = "codec.col";

/// The `[elements]` workspaces: the largest stream (the last block's output) and column buffer.
pub fn workspaces() -> [(&'static str, usize); 3] {
    let stream = MAX_FRAMES * SAMPLES_PER_FRAME * OUT_CHANNELS;
    [(STREAM, stream), (BRANCH, stream), (COLUMNS, stream * TAPS)]
}

/// A var expression for `frames · m` rows.
fn rows(m: usize) -> Value {
    if m == 1 { json!("frames") } else { json!({"mul": ["frames", m]}) }
}

/// An elementwise launch over `groups` eight-channel groups per frame.
fn each(g: &mut Gen, label: &str, entry: &str, groups: usize, mut args: Vec<(&str, Value)>) {
    let total = rows(groups);
    args.push(("i32", count(&total)));
    g.launch(label, entry, [json!({"ceil_div": [total, THREADS]}), json!(1), json!(1)], THREADS, args);
}

struct Snake {
    alpha: String,
    inv: String,
}

impl Snake {
    fn load(g: &mut Gen, file: &File, key: &str, name: &str, channels: usize) -> Result<Self> {
        let alpha = file.expect(&format!("{PREFIX}acoustic_decoder.{key}.alpha"), &[1, channels, 1])?;
        Ok(Self {
            alpha: g.weight_f32(&format!("codec.{name}.alpha"), alpha.data.clone()),
            inv: g.weight_f32(&format!("codec.{name}.inv"), alpha.data.iter().map(|a| 1.0 / (a + 1e-9)).collect()),
        })
    }
}

/// A conv's tap matrix and bias.
struct Conv {
    taps: String,
    bias: String,
}

impl Conv {
    fn load(g: &mut Gen, file: &File, key: &str, name: &str, shape: [usize; 3]) -> Result<Self> {
        let w = file.expect(&format!("{PREFIX}acoustic_decoder.{key}.weight"), &shape)?;
        let b = file.expect(&format!("{PREFIX}acoustic_decoder.{key}.bias"), &shape[..1])?;
        Ok(Self {
            taps: g.weight(&format!("codec.{name}.w"), &[shape[0], shape[2] * shape[1]], &conv_taps(&w)),
            bias: g.weight(&format!("codec.{name}.b"), &[shape[0]], &b.data),
        })
    }
}

/// `y = im2col(act(x)) · wᵀ` for a 7-tap conv dilated by `dil` over `m · frames` rows, `cin`
/// channels in and `cout` out; the conv's bias is left to its consumer.
fn conv(
    g: &mut Gen,
    label: &str,
    (x, y): (&str, &str),
    act: Option<&Snake>,
    (w, m, dil): (&str, usize, usize),
    (cin, cout): (usize, usize),
) {
    let mut args = vec![inb(x)];
    let entry = match act {
        Some(s) => {
            args.extend([inf(&s.alpha), inf(&s.inv)]);
            "codec_im2col_snake"
        }
        None => "codec_im2col",
    };
    args.extend([outb(COLUMNS), ("i32", count(&rows(m))), i32a(cin), i32a(dil)]);
    each(g, &format!("{label}.im2col"), entry, m * TAPS * (cin / 8), args);
    g.gemm_rows(&format!("{label}.gemm"), (buf(y), buf(COLUMNS), buf(w)), count(&rows(m)), (cout, TAPS * cin));
}

/// Emits the decoder's calls into `g`: `frame_codes` `[frames, 8]` in, `pcm` out.
pub fn build(g: &mut Gen, file: &File) -> Result<()> {
    ensure!(STRIDES.iter().product::<usize>() == SAMPLES_PER_FRAME, "the strides do not make a frame's samples");
    let p = |s: String| format!("{PREFIX}{s}");
    let embeds = (0..CODEBOOKS)
        .map(|q| file.expect(&p(format!("quantizer.quantizers.{q}.codebook.embed")), &[CODES, WIDTH]))
        .collect::<Result<Vec<_>>>()?;
    let projects = (0..CODEBOOKS)
        .map(|q| {
            let key = |s: &str| p(format!("quantizer.quantizers.{q}.project_out.{s}"));
            Ok((file.expect(&key("weight"), &[LATENT, WIDTH])?, file.expect(&key("bias"), &[LATENT])?))
        })
        .collect::<Result<Vec<_>>>()?;
    let quantizers: Vec<Quantizer> =
        embeds.iter().zip(&projects).map(|(embed, (w, b))| Quantizer { embed, project_out: (w, b) }).collect();
    let fc2 =
        (file.expect(&p("fc2.weight".into()), &[ACOUSTIC, LATENT])?, file.expect(&p("fc2.bias".into()), &[ACOUSTIC])?);
    let folded = fold(&quantizers, &Fc2(&fc2.0, &fc2.1));
    let tables = g.weight("codec.tables", &[CODEBOOKS * CODES, ACOUSTIC], &folded.tables);
    let bias = g.weight("codec.bias", &[ACOUSTIC], &folded.bias);
    each(
        g,
        "codec.lookup",
        "codec_lookup",
        ACOUSTIC / 8,
        vec![ini("frame_codes"), inb(&tables), inb(&bias), outb(STREAM), i32a(ACOUSTIC), i32a(CODES)],
    );

    let first = Conv::load(g, file, "conv1", "conv1", [LATENT, ACOUSTIC, TAPS])?;
    conv(g, "codec.conv1", (STREAM, BRANCH), None, (&first.taps, 1, 1), (ACOUSTIC, LATENT));
    let mut m = 1;
    for (i, &stride) in STRIDES.iter().enumerate() {
        let (cin, cout) = (LATENT >> i, LATENT >> (i + 1));
        let label = format!("codec.b{i}");
        let snake = Snake::load(g, file, &format!("block.{i}.snake1"), &format!("b{i}.snake"), cin)?;
        let (alpha, inv) = (inf(&snake.alpha), inf(&snake.inv));
        if i == 0 {
            let args = vec![inb(BRANCH), inb(&first.bias), alpha, inv, outb(BRANCH), i32a(cin)];
            each(g, &format!("{label}.snake"), "codec_bias_snake", m * cin / 8, args);
        } else {
            each(
                g,
                &format!("{label}.snake"),
                "codec_snake",
                m * cin / 8,
                vec![inb(STREAM), alpha, inv, outb(BRANCH), i32a(cin)],
            );
        }
        let key = format!("block.{i}.conv_t1");
        let up = file.expect(&p(format!("acoustic_decoder.{key}.weight")), &[cin, cout, 2 * stride])?;
        let up_b = file.expect(&p(format!("acoustic_decoder.{key}.bias")), &[cout])?;
        let up_w = g.weight(&format!("codec.b{i}.up.w"), &[2 * stride * cout, cin], &transposed_taps(&up));
        let up_b = g.weight(&format!("codec.b{i}.up.b"), &[cout], &up_b.data);
        g.gemm_rows(
            &format!("{label}.up"),
            (buf(COLUMNS), buf(BRANCH), buf(&up_w)),
            count(&rows(m)),
            (2 * stride * cout, cin),
        );
        let args = vec![
            inb(COLUMNS),
            inb(&up_b),
            outb(STREAM),
            ("i32", count(&rows(m))),
            i32a(stride),
            i32a(stride.div_ceil(2)),
            i32a(cout),
        ];
        m *= stride;
        each(g, &format!("{label}.col2im"), "codec_col2im", m * cout / 8, args);

        for (u, &dil) in DILATIONS.iter().enumerate() {
            let name = format!("b{i}.r{u}");
            let unit = format!("block.{i}.res_unit{}", u + 1);
            let s1 = Snake::load(g, file, &format!("{unit}.snake1"), &format!("{name}.s1"), cout)?;
            let c1 = Conv::load(g, file, &format!("{unit}.conv1"), &format!("{name}.c1"), [cout, cout, TAPS])?;
            let s2 = Snake::load(g, file, &format!("{unit}.snake2"), &format!("{name}.s2"), cout)?;
            let c2 = Conv::load(g, file, &format!("{unit}.conv2"), &format!("{name}.c2"), [cout, cout, 1])?;
            let label = format!("codec.{name}");
            conv(g, &format!("{label}.conv1"), (STREAM, BRANCH), Some(&s1), (&c1.taps, m, dil), (cout, cout));
            let args = vec![inb(BRANCH), inb(&c1.bias), inf(&s2.alpha), inf(&s2.inv), outb(BRANCH), i32a(cout)];
            each(g, &format!("{label}.snake"), "codec_bias_snake", m * cout / 8, args);
            g.gemm_rows(
                &format!("{label}.conv2"),
                (buf(COLUMNS), buf(BRANCH), buf(&c2.taps)),
                count(&rows(m)),
                (cout, cout),
            );
            let args = vec![io(STREAM), inb(COLUMNS), inb(&c2.bias), i32a(cout)];
            each(g, &format!("{label}.residual"), "codec_residual", m * cout / 8, args);
        }
    }

    ensure!(m == SAMPLES_PER_FRAME, "the blocks end at {m} rows per frame");
    let last = Snake::load(g, file, "snake1", "out.snake", OUT_CHANNELS)?;
    let w = file.expect(&p("acoustic_decoder.conv2.weight".into()), &[1, OUT_CHANNELS, TAPS])?;
    let b = file.expect(&p("acoustic_decoder.conv2.bias".into()), &[1])?;
    let w: Vec<f32> =
        (0..TAPS).flat_map(|j| (0..OUT_CHANNELS).map(move |c| (j, c))).map(|(j, c)| w.data[c * TAPS + j]).collect();
    let (w, b) = (g.weight_f32("codec.out.w", w), g.weight_f32("codec.out.b", b.data));
    let total = rows(m);
    g.launch(
        "codec.out",
        "codec_conv_out",
        [json!({"ceil_div": [total, 256]}), json!(1), json!(1)],
        256,
        vec![
            inb(STREAM),
            inf(&last.alpha),
            inf(&last.inv),
            inf(&w),
            inf(&b),
            ("out buffer<f32>", buf("pcm")),
            ("i32", count(&total)),
            i32a(OUT_CHANNELS),
        ],
    );
    Ok(())
}
