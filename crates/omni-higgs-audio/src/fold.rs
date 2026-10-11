//! The codec's quantizer folded into lookup tables.
//!
//! Decoding a frame sums, over the eight quantizers, a codebook entry (64 wide) projected to 1024
//! channels, then maps the sum through `fc2` to the 256 channels the acoustic decoder reads. Every
//! step after the lookup is linear, so each codebook folds with its projection and `fc2` into a
//! table of 256-wide rows, and a frame's input is the sum of eight rows plus one constant.

use omni_kern::weights::Host;

/// One quantizer's tensors.
pub struct Quantizer<'a> {
    /// `[codes, width]`.
    pub embed: &'a Host,
    /// `[channels, width]` and `[channels]`.
    pub project_out: (&'a Host, &'a Host),
}

/// `fc2`'s `[out, channels]` weight and `[out]` bias.
pub struct Fc2<'a>(pub &'a Host, pub &'a Host);

/// The quantizers' tables, `[quantizer][code][out]`, and the constant every frame adds.
pub struct Folded {
    pub tables: Vec<f32>,
    pub bias: Vec<f32>,
}

pub fn fold(quantizers: &[Quantizer], fc2: &Fc2) -> Folded {
    let Fc2(w, b) = fc2;
    let (out, channels) = (w.shape[0], w.shape[1]);
    let mut bias = b.data.clone();
    let mut tables = Vec::new();
    for q in quantizers {
        let (codes, width) = (q.embed.shape[0], q.embed.shape[1]);
        let (po, pb) = q.project_out;
        // fc2 · project_out: [out, width].
        let mut through = vec![0f32; out * width];
        for (o, row) in through.chunks_mut(width).enumerate() {
            let weights = &w.data[o * channels..(o + 1) * channels];
            for (c, &a) in weights.iter().enumerate() {
                for (acc, &p) in row.iter_mut().zip(&po.data[c * width..(c + 1) * width]) {
                    *acc += a * p;
                }
            }
            bias[o] += weights.iter().zip(&pb.data).map(|(a, b)| a * b).sum::<f32>();
        }
        for code in 0..codes {
            let e = &q.embed.data[code * width..(code + 1) * width];
            tables.extend(through.chunks(width).map(|row| row.iter().zip(e).map(|(a, b)| a * b).sum::<f32>()));
        }
    }
    Folded { tables, bias }
}
